import os
from collections import OrderedDict

import numpy as np
import torch
from torch import nn

from network.blocks import MLP, MultiHeadAttnLayer, build_state_encoder
from network.utils.helper import LossUtils, EnergyUtils, TaskSelectionUtils


def get_parameter_count(model):
    total_num = 0
    for p in model.parameters():
        shape = p.shape
        num = 1
        for d in shape:
            num *= d
        total_num += num
    return total_num


class EffectPrediction(nn.Module):
    def __init__(self, seed, config):
        super(EffectPrediction, self).__init__()
        self.seed = seed
        self.model_name = config["name"]
        self.device = torch.device(config["device"])
        self.criterion = torch.nn.L1Loss()
        self.task_ids = np.arange(0, config["num_tasks"])
        self.proj_ = "cnn" not in config
        self.encoder = self.build_encoder(config).to(self.device)
        self.decoder = self.build_decoder(config).to(self.device)
        self.optimizer = torch.optim.AdamW(lr=config["learning_rate"],
                                           params=[
                                               {"params": self.encoder.parameters()},
                                               {"params": self.decoder.parameters()}],
                                           amsgrad=True)
        self.freeze_all()
        self.num_epochs = config["epoch"]
        self.loss_util = LossUtils(config["num_tasks"])
        self.energy_util = EnergyUtils(config["num_tasks"])
        self.selection_util = TaskSelectionUtils(config["num_tasks"], config["selection"],
                                                 config["task_sequence"])
        self.selection_type = config["selection"]
        self.model_type = config["mode"]

        self.save_path = config["save"] + "/model_ckpts"
        self.result_path = config["save"] + "/plots/"

    def build_encoder(self, config):
        raise NotImplementedError

    def build_decoder(self, config):
        raise NotImplementedError

    def _train_mode(self, task_id, train_mode=True):
        raise NotImplementedError

    def one_pass_others(self, winner, loader):
        raise NotImplementedError

    def get_next(self):
        raise NotImplementedError

    def freeze_task(self, task_id, unfreeze=False):
        raise NotImplementedError

    def freeze_shared(self, unfreeze=False):
        raise NotImplementedError

    def loss(self, task_id, state, effect, action):
        raise NotImplementedError

    def forward_mb(self, task_id, sample_state, sample_effect, sample_action):
        if self.proj_:
            state = sample_state.to(self.device)
        else:
            state = sample_state.unsqueeze(1).to(self.device)  # when input is img convert shape into (N, C, H, W)
        action = sample_action.to(self.device)
        effect = sample_effect.to(self.device)
        loss = self.loss(task_id, state, effect, action)
        return loss

    def one_epoch_optimize(self, task_id, loader):
        self._train_mode(task_id)
        running_loss = 0.0
        energy_usage = 0.
        for (state, effect, action) in loader:
            self.optimizer.zero_grad()
            loss = self.forward_mb(task_id, state, effect, action)
            loss.backward()
            self.optimizer.step()
            energy_usage += self.get_synaptic_cost(task_id) + self.sum_dense_activations(task_id)
            running_loss += loss.item()

        avg_loss = running_loss / len(loader)
        avg_energy = energy_usage / len(loader)
        self.loss_util.update_task_loss(avg_loss, task_id)
        self.energy_util.update_task_energy(avg_energy, task_id)
        return avg_loss

    def evaluate_epoch(self, task_id, loader, val_=True):
        self._train_mode(task_id, train_mode=False)
        running_vloss = 0.0
        energy_usage = 0.
        with torch.no_grad():
            for (state, effect, action) in loader:
                loss = self.forward_mb(task_id, state, effect, action)
                running_vloss += loss.item()
                energy_usage += self.sum_dense_activations(task_id)
            avg_vloss = running_vloss / len(loader)
            avg_energy = energy_usage / len(loader)
            if val_:
                self.loss_util.update_task_loss(avg_vloss, task_id, is_eval=True)
                self.energy_util.update_task_energy(avg_energy, task_id, is_eval=True)
            else:
                self.loss_util.update_last_loss(avg_vloss, task_id)
                self.energy_util.update_last_energy(avg_energy, task_id)
            return avg_vloss

    def evaluate_all(self, loaders):
        eval_loss = np.zeros(len(self.task_ids))
        for i, t in enumerate(self.task_ids):
            eval_loss[i] = self.evaluate_epoch(t, loaders[t])
        return np.mean(eval_loss)

    def pre_train(self, task_id, tr_loader, val_loader, count=10):
        self.freeze_task(task_id, unfreeze=True)
        for _ in range(count):
            self.one_epoch_optimize(task_id, tr_loader)
            self.evaluate_epoch(task_id, val_loader)
        self.freeze_task(task_id, unfreeze=False)

        # Record pre-train results to the plot
        self.loss_util.update_loss_plot(copy=True, task_id=task_id)
        self.energy_util.update_energy_plot(copy=True, task_id=task_id)

    def train_(self, train_loaders, val_loaders):
        for t in self.task_ids:
            self.pre_train(task_id=t, tr_loader=train_loaders[t], val_loader=val_loaders[t])

        winner_id = self.get_next()
        best_loss = 1e6
        self.freeze_shared(unfreeze=True)  # for multitask models, this is required, others pass
        for e in range(1, self.num_epochs + 1):
            self.freeze_task(task_id=winner_id, unfreeze=True)
            self.one_epoch_optimize(task_id=winner_id, loader=train_loaders[winner_id])
            self.freeze_task(task_id=winner_id)

            # for interleaved multitask models, this is required, others pass
            self.one_pass_others(winner=winner_id, loader=train_loaders)

            self.loss_util.update_loss_plot()
            self.energy_util.update_energy_plot()
            avg_vloss = self.evaluate_all(val_loaders)

            if avg_vloss < best_loss:  # avg task loss
                best_loss = avg_vloss
                # self.save(self.save_path, "_best")

            if e == self.num_epochs:
                print("------- Seed {} finished training!".format(self.seed))
                # self.save(self.save_path, "_last")

            if e % 200 == 0:
                np.save('{}/train-energy-bar-epoch-{}-seed-{}.npy'.format(self.result_path, e, self.seed),
                        np.asarray(self.energy_util.get_total_energy()))
                np.save('{}/eval-energy-bar-epoch-{}-seed-{}.npy'.format(self.result_path, e, self.seed),
                        np.asarray(self.energy_util.get_total_energy(is_eval=True)))

            winner_id = self.get_next()

        # Save final results
        np.save('{}/train-loss-plot-seed-{}.npy'.format(self.result_path, self.seed),
                np.asarray(self.loss_util.train_loss_history_plot))
        np.save('{}/eval-loss-plot-seed-{}.npy'.format(self.result_path, self.seed),
                np.asarray(self.loss_util.eval_loss_history_plot))

    def freeze_all(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False
        self.eval()

    def unfreeze_all(self):
        for param in self.encoder.parameters():
            param.requires_grad = True
        for param in self.decoder.parameters():
            param.requires_grad = True
        self.train()

    def load(self, path, ext):
        encoder = self.encoder
        decoder = self.decoder

        encoder_dict = torch.load(os.path.join(path, "seed_" + str(self.seed) + "_encoder" + ext + ".ckpt"))
        decoder_dict = torch.load(os.path.join(path, "seed_" + str(self.seed) + "_decoder" + ext + ".ckpt"))
        encoder.load_state_dict(encoder_dict)
        decoder.load_state_dict(decoder_dict)

    def save(self, path, ext):
        encoder = self.encoder
        decoder = self.decoder

        encoder_dict = encoder.eval().cpu().state_dict()
        decoder_dict = decoder.eval().cpu().state_dict()
        torch.save(encoder_dict, os.path.join(path, "seed_" + str(self.seed) + "_encoder" + ext + ".ckpt"))
        torch.save(decoder_dict, os.path.join(path, "seed_" + str(self.seed) + "_decoder" + ext + ".ckpt"))
        encoder.train().to(self.device)
        decoder.train().to(self.device)

    def print_model(self):
        encoder = self.encoder
        decoder = self.decoder
        # print("=" * 10 + "ENCODER" + "=" * 10)
        # print(encoder)
        # print("parameter count: %d" % utils.get_parameter_count(encoder))
        # print("=" * 27)
        # print("=" * 10 + "DECODER" + "=" * 10)
        # print(decoder)
        # print("parameter count: %d" % utils.get_parameter_count(decoder))
        # print("=" * 27)

    def sum_dense_activations(self, task_id):
        total_activation_sum = (self.sum_dense_activations_aux(self.encoder[task_id]) +
                                self.sum_dense_activations_aux(self.decoder[task_id]))
        return total_activation_sum

    def sum_dense_activations_aux(self, model):
        total_activation_sum = 0
        for module in model.children():
            if hasattr(module, 'activation_out'):
                if module.activation_out is not None:
                    total_activation_sum += module.activation_out
            else:
                total_activation_sum += self.sum_dense_activations_aux(module)  # Recursive call
        return total_activation_sum

    def get_synaptic_cost(self, task_id):
        synaptic_transmission_cost = 0
        for param in self.encoder[task_id].parameters():
            if param.requires_grad:
                weight = param.data.detach().cpu().numpy()
                gradient = param.grad.detach().cpu().numpy()
                # Element-wise multiplication of weights and gradients
                cost = np.sum(np.abs(weight) * np.abs(gradient))
                synaptic_transmission_cost += cost
        for param in self.decoder[task_id].parameters():
            if param.requires_grad:
                weight = param.data.detach().cpu().numpy()
                gradient = param.grad.detach().cpu().numpy()
                # Element-wise multiplication of weights and gradients
                cost = np.sum(np.abs(weight) * np.abs(gradient))
                synaptic_transmission_cost += cost
        return synaptic_transmission_cost


class SingleTask(EffectPrediction):
    def __init__(self, seed, config):
        super(SingleTask, self).__init__(seed, config)

    def build_encoder(self, config):
        num_tasks = config["num_tasks"]
        task_encoders = nn.ModuleList()
        for n in range(num_tasks):
            state_encoder = build_state_encoder(config=config, task_idx=n)
            action_proj = MLP(layer_info=[config["action_size"][n], config["rep_action"]])
            attn_layer = MultiHeadAttnLayer(embed_dim=config["rep_state"] + config["rep_action"],
                                            num_heads=config["num_heads"])
            encoder = nn.Sequential(OrderedDict([("state_encoder", state_encoder),
                                                 ("action_proj", action_proj),
                                                 ("attn_layer", attn_layer)]))
            task_encoders.append(encoder)
        return task_encoders

    def build_decoder(self, config):
        num_tasks = config["num_tasks"]
        task_decoders = nn.ModuleList()
        for n in range(num_tasks):
            effect_decoder = MLP(
                layer_info=[config["rep_state"] + config["rep_action"]] +
                           [config["hidden_dim"]] * config["dec_depth_effect"] +
                           [config["out_size"][n]],
                batch_norm=config["batch_norm"])
            decoder = nn.Sequential(OrderedDict([("effect_decoder", effect_decoder)]))
            task_decoders.append(decoder)
        return task_decoders

    def _train_mode(self, task_id, train_mode=True):
        if train_mode:
            self.encoder[task_id].train()
            self.decoder[task_id].train()
        else:
            self.encoder[task_id].eval()
            self.decoder[task_id].eval()

    def loss(self, task_id, state, effect, action):
        state_code = self.encoder[task_id].state_encoder(state)
        action_code = self.encoder[task_id].action_proj(action)
        action_state = torch.cat([state_code, action_code], dim=-1).unsqueeze(0)
        keys = action_state.clone()
        values = action_state.clone()
        query = action_state.clone()
        h_attn = self.encoder[task_id].attn_layer(query, keys, values).squeeze(0)
        effect_pred = self.decoder[task_id](h_attn)
        return self.criterion(effect_pred, effect)

    def one_pass_others(self, winner, loader):
        pass

    def get_next(self):
        return self.selection_util.get_winner()

    def freeze_task(self, task_id, unfreeze=False):
        for param in self.encoder[task_id].parameters():
            param.requires_grad = unfreeze
        for param in self.decoder[task_id].parameters():
            param.requires_grad = unfreeze

    def freeze_shared(self, unfreeze=False):
        pass


class MultiTask(EffectPrediction):
    def __init__(self, seed, config):
        super(MultiTask, self).__init__(seed, config)

    def build_encoder(self, config):
        num_tasks = config["num_tasks"]
        task_encoders = nn.ModuleList()

        # Shared encoder and multi-head attention layer
        state_encoder = build_state_encoder(config=config, shared=True)
        attn_layer = MultiHeadAttnLayer(embed_dim=config["rep_state"] + 1,  # +1 for flag
                                        num_heads=config["num_heads"])
        for n in range(num_tasks):
            sub_encoder = MLP(layer_info=[config["hidden_dim"]] * config["enc_depth_sub"] + [config["rep_state"]],
                              batch_norm=config["batch_norm"])
            action_proj = MLP(layer_info=[config["action_size"][n], config["rep_action"]])
            encoder_dict = OrderedDict([("state_encoder", state_encoder),  # shared
                                        ("sub_encoder", sub_encoder),
                                        ("action_proj", action_proj),
                                        ("attn_layer", attn_layer)])
            if self.proj_:
                proj_layer = MLP(layer_info=[config["in_size"][n], config["hidden_dim"]])
                encoder_dict.update({"proj_layer": proj_layer})
                encoder_dict.move_to_end("proj_layer", last=False)

            encoder = nn.Sequential(encoder_dict)
            task_encoders.append(encoder)
        return task_encoders

    def build_decoder(self, config):
        num_tasks = config["num_tasks"]
        task_decoders = nn.ModuleList()
        for n in range(num_tasks):
            effect_decoder = MLP(layer_info=[config["rep_state"] + config["rep_action"] + 1] +
                                            [config["hidden_dim"]] * config["dec_depth_effect"] +
                                            [config["out_size"][n]],
                                 batch_norm=config["batch_norm"])
            decoder = nn.Sequential(OrderedDict([("effect_decoder", effect_decoder)]))
            task_decoders.append(decoder)
        return task_decoders

    def _train_mode(self, task_id, train_mode=True):  # Task specific parts only
        if train_mode:
            if self.proj_:
                self.encoder[task_id].proj_layer.train()
            self.encoder[task_id].sub_encoder.train()
            self.encoder[task_id].action_proj.train()
            self.decoder[task_id].train()
        else:
            if self.proj_:
                self.encoder[task_id].proj_layer.eval()
            self.encoder[task_id].sub_encoder.eval()
            self.encoder[task_id].action_proj.eval()
            self.decoder[task_id].eval()

    def loss(self, task_id, state, effect, action):
        if self.proj_:
            state = self.encoder[task_id].proj_layer(state)
        state_code = self.encoder[task_id].state_encoder(state)  # shared encoder
        task_reprs = []
        for idx in self.task_ids:
            if idx != task_id:
                with torch.no_grad():  # Frozen, no gradients computed
                    task_state_code = self.encoder[idx].sub_encoder(state_code)
                    flag = torch.FloatTensor(torch.zeros([task_state_code.shape[0], 1])).to(self.device)
                    rep = torch.cat([task_state_code, flag], dim=-1)
            else:  # trainable
                task_state_code = self.encoder[idx].sub_encoder(state_code)
                flag = torch.FloatTensor(torch.ones([task_state_code.shape[0], 1])).to(self.device)
                rep = torch.cat([task_state_code, flag], dim=-1)
            task_reprs.append(rep.unsqueeze(0))  # rep shape: (1, batch_size, (1 + rep_state))

        keys = torch.cat(task_reprs, dim=0)  # Shape: (num_tasks, batch_size, (1 + rep_state))
        values = keys.clone()

        # Query is from current task's representation
        query = task_reprs[task_id].clone()  # query shape: (1, batch_size, (1 + rep_state))
        h_attn = self.encoder[task_id].attn_layer(query, keys, values).squeeze(
            0)  # Shape: (batch_size, (1 + rep_dim))

        task_action_code = self.encoder[task_id].action_proj(action)
        effect_pred = self.decoder[task_id](torch.hstack((h_attn, task_action_code)))
        return self.criterion(effect_pred, effect)

    def one_pass_others(self, winner, loader):
        for t in self.task_ids:
            if t != winner:
                self.evaluate_epoch(t, loader[t], val_=False)  # Do not train, just get training loss

    def get_next(self):
        if "lp" in self.selection_type:
            for t in self.task_ids:
                self.selection_util.calculate_lp(loss=self.loss_util.train_loss_history[t], index=t)
            self.selection_util.save_lp()

        if "e" in self.selection_type:
            for t in self.task_ids:
                self.selection_util.calculate_ep(energy=self.energy_util.train_energy_history[t], index=t)
            self.selection_util.save_ep()

        return self.selection_util.get_winner()

    def freeze_task(self, task_id, unfreeze=False):  # Only freeze/unfreeze task specific parts.
        if self.proj_:
            for param in self.encoder[task_id].proj_layer.parameters():
                param.requires_grad = unfreeze
        for param in self.encoder[task_id].sub_encoder.parameters():
            param.requires_grad = unfreeze
        for param in self.encoder[task_id].action_proj.parameters():
            param.requires_grad = unfreeze
        for param in self.decoder[task_id].parameters():
            param.requires_grad = unfreeze

    def freeze_shared(self, unfreeze=False):
        for param in self.encoder[0].state_encoder.parameters():  # shared state encoder
            param.requires_grad = unfreeze
        for param in self.encoder[0].attn_layer.parameters():  # shared attention layer
            param.requires_grad = unfreeze
        if unfreeze:
            self.encoder[0].state_encoder.train()
            self.encoder[0].attn_layer.train()
        else:
            self.encoder[0].state_encoder.eval()
            self.encoder[0].attn_layer.eval()

# class BlockedMultiTask(MultiTask):
#     def __init__(self, seed, config):
#         super(BlockedMultiTask, self).__init__(seed, config)
#
#     def train_(self, train_loaders, val_loaders):
#         winner_id = self.get_next()
#         self.freeze_shared(unfreeze=True)  # for multitask models, this is required, others pass
#         self.freeze_task(task_id=winner_id, unfreeze=True)
#         trained_tasks = []
#         for e in range(1, self.num_epochs + 1):
#             self.one_epoch_optimize(task_id=winner_id, loader=train_loaders[winner_id])
#             self.evaluate_epoch(task_id=winner_id, loader=val_loaders[winner_id])
#             for idx in trained_tasks:
#                 self.evaluate_epoch(task_id=idx, loader=train_loaders[idx], val_=False)
#                 self.evaluate_epoch(task_id=idx, loader=val_loaders[idx])
#             next_winner = self.get_next()
#             if winner_id != next_winner:
#                 self.freeze_task(task_id=winner_id)
#                 self.freeze_task(task_id=next_winner, unfreeze=True)
#                 trained_tasks.append(winner_id)
#             winner_id = next_winner
#             self.loss_util.update_loss_plot()
#             self.energy_util.update_energy_plot()
