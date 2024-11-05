import os
from collections import OrderedDict

import numpy as np
import torch
from torch import nn

from network.blocks import MLP, Linear, MultiHeadAttnLayer
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

        # Create subdir for model checkpoints and results
        if not os.path.exists(config["save"] + "/model_ckpts"):
            os.makedirs(config["save"] + "/model_ckpts")

        self.save_path = config["save"] + "/model_ckpts"

        if not os.path.exists(config["save"] + "/plots"):
            os.makedirs(config["save"] + "/plots")

        self.result_path = config["save"] + "/plots/"

    def build_encoder(self, config):
        raise NotImplementedError

    def build_decoder(self, config):
        raise NotImplementedError

    def _train_mode(self, task_id, train_mode=True):
        raise NotImplementedError

    # Function to pad input to match the required dimension
    def pad_input(self, input_data):
        raise NotImplementedError

    def one_pass_others(self, winner, loader):
        raise NotImplementedError

    def get_next(self):
        raise NotImplementedError

    def freeze_task(self, task_id, unfreeze=False):
        raise NotImplementedError

    def freeze_lvl1(self, unfreeze=False):
        raise NotImplementedError

    def loss(self, task_id, state, effect, action):
        raise NotImplementedError

    def forward_mb(self, task_id, sample_state, sample_effect, sample_action):
        state_pad = self.pad_input(input_data=sample_state)  # pad with zeros if needed
        state = state_pad.to(self.device)
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

    def train_(self, train_loaders, val_loaders):
        for t in self.task_ids:
            self.pre_train(task_id=t, tr_loader=train_loaders[t], val_loader=val_loaders[t])

        winner_id = self.get_next()
        best_loss = 1e6
        self.freeze_lvl1(unfreeze=True)  # for multitask models, this is required, others pass
        for e in range(1, self.num_epochs + 1):
            self.freeze_task(task_id=winner_id, unfreeze=True)
            self.one_epoch_optimize(task_id=winner_id, loader=train_loaders[winner_id])
            self.freeze_task(task_id=winner_id)

            # for interleaved multitask models, this is required, others pass
            self.one_pass_others(winner=winner_id, loader=train_loaders)

            self.loss_util.update_loss_plot()
            avg_vloss = self.evaluate_all(val_loaders)

            if avg_vloss < best_loss:  # avg task loss
                best_loss = avg_vloss
                self.save(self.save_path, "_best")

            if e == self.num_epochs:
                self.save(self.save_path, "_last")

            if e % 200 == 0:
                np.save('{}/train-energy-bar-epoch-{}-seed-{}.npy'.format(self.result_path, e, self.seed),
                        np.asarray(self.energy_util.get_total_energy()))
                np.save('{}/test-energy-bar-epoch-{}-seed-{}.npy'.format(self.result_path, e, self.seed),
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
        return total_activation_sum / 1e3

    def sum_dense_activations_aux(self, model):
        total_activation_sum = 0
        for module in model.children():
            if isinstance(module, Linear):
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
            lvl1encoder = MLP(layer_info=[config["in_size"][n]] + [config["hidden_dim"]] * config["enc_depth_shared"],
                              batch_norm=config["batch_norm"])
            lvl2encoder = MLP(layer_info=[config["hidden_dim"]] * config["enc_depth_task"] + [config["rep_dim"]],
                              batch_norm=config["batch_norm"])
            encoder = nn.Sequential(OrderedDict([("lvl1encoder", lvl1encoder),
                                                 ("lvl2encoder", lvl2encoder)]))
            task_encoders.append(encoder)
        return task_encoders

    def build_decoder(self, config):
        num_tasks = config["num_tasks"]
        task_decoders = nn.ModuleList()
        for n in range(num_tasks):
            lvl1decoder = MLP(layer_info=[config["rep_dim"] + config["action_dim"]] + [config["hidden_dim"]] * config[
                "dec_depth_shared"],
                              batch_norm=config["batch_norm"])
            lvl2decoder = MLP(
                layer_info=[config["hidden_dim"]] * config["dec_depth_task"] + [config["out_size"][n]],
                batch_norm=config["batch_norm"])
            decoder = nn.Sequential(OrderedDict([("lvl1decoder", lvl1decoder),
                                                 ("lvl2decoder", lvl2decoder)]))
            task_decoders.append(decoder)
        return task_decoders

    def pad_input(self, input_data):
        return input_data

    def _train_mode(self, task_id, train_mode=True):
        if train_mode:
            self.encoder[task_id].train()
            self.decoder[task_id].train()
        else:
            self.encoder[task_id].eval()
            self.decoder[task_id].eval()

    def loss(self, task_id, state, effect, action):
        h = self.encoder[task_id](state)
        h_aug = torch.cat([h, action], dim=-1)
        effect_pred = self.decoder[task_id](h_aug)
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

    def freeze_lvl1(self, unfreeze=False):
        pass


class MultiTask(EffectPrediction):
    def __init__(self, seed, config):
        super(MultiTask, self).__init__(seed, config)
        self.input_dim = max(config["in_size"])

    def build_encoder(self, config):
        num_tasks = config["num_tasks"]
        task_encoders = nn.ModuleList()
        in_size = max(config["in_size"])
        # Shared encoder
        lvl1encoder = MLP(layer_info=[in_size] + [config["hidden_dim"]] * config["enc_depth_shared"],
                          batch_norm=config["batch_norm"])

        # Shared attention layer
        attn_layer = MultiHeadAttnLayer(embed_dim=config["rep_dim"])
        for n in range(num_tasks):
            lvl2encoder = MLP(layer_info=[config["hidden_dim"]] * config["enc_depth_task"] + [config["rep_dim"]],
                              batch_norm=config["batch_norm"])
            encoder = nn.Sequential(OrderedDict([("lvl1encoder", lvl1encoder),
                                                 ("lvl2encoder", lvl2encoder),
                                                 ("attn_layer", attn_layer)]))
            task_encoders.append(encoder)
        return task_encoders

    def build_decoder(self, config):
        num_tasks = config["num_tasks"]
        task_decoders = nn.ModuleList()
        # Shared decoder
        lvl1decoder = MLP(layer_info=[config["rep_dim"] + config["action_dim"]] + [config["hidden_dim"]] * config[
            "dec_depth_shared"],
                          batch_norm=config["batch_norm"])
        for n in range(num_tasks):
            lvl2decoder = MLP(
                layer_info=[config["hidden_dim"]] * config["dec_depth_task"] + [config["out_size"][n]],
                batch_norm=config["batch_norm"])
            decoder = nn.Sequential(OrderedDict([("lvl1decoder", lvl1decoder),
                                                 ("lvl2decoder", lvl2decoder)]))
            task_decoders.append(decoder)
        return task_decoders

    def pad_input(self, input_data):
        current_dim = input_data.size(1)
        if current_dim < self.input_dim:
            padding_size = self.input_dim - current_dim
            padded_input = torch.cat([input_data, torch.zeros(input_data.size(0), padding_size)], dim=1)
        else:
            padded_input = input_data
        return padded_input

    def _train_mode(self, task_id, train_mode=True):
        if train_mode:
            self.encoder[task_id].lvl2encoder.train()
            self.decoder[task_id].lvl2decoder.train()
        else:
            self.encoder[task_id].lvl2encoder.eval()
            self.decoder[task_id].lvl2decoder.eval()

    def loss(self, task_id, state, effect, action):
        lvl1_code = self.encoder[task_id].lvl1encoder(state)    # shared encoder
        lvl2_codes = []
        for idx in self.task_ids:
            if idx != task_id:
                with torch.no_grad():     # Frozen, no gradients computed
                    task_code = self.encoder[idx].lvl2encoder(lvl1_code)  # task specific encoder
            else:   # trainable
                task_code = self.encoder[idx].lvl2encoder(lvl1_code)
            lvl2_codes.append(task_code.unsqueeze(0))   # Shape: (1, batch_size, rep_dim)

        keys = torch.cat(lvl2_codes, dim=0)  # Shape: (num_tasks, batch_size, rep_dim)
        values = keys.clone()

        # Query is from current task's representation
        query = lvl2_codes[task_id]  # Shape: (1, batch_size, rep_dim)
        h = self.encoder[task_id].attn_layer(query, keys, values).squeeze(0)     # Shape: (batch_size, rep_dim)
        h_aug = torch.cat([h, action], dim=-1)
        effect_pred = self.decoder[task_id](h_aug)
        return self.criterion(effect_pred, effect)

    def one_pass_others(self, winner, loader):
        for t in self.task_ids:
            if t != winner:
                self.evaluate_epoch(t, loader[t], val_=False)  # Do not train, just get training loss

    def get_next(self):
        if self.selection_type == "lp":
            for t in self.task_ids:
                self.selection_util.calculate_lp(loss=self.loss_util.train_loss_history[t], index=t)
            self.selection_util.save_progress()

        return self.selection_util.get_winner()

    def freeze_task(self, task_id, unfreeze=False):  # Only freeze/unfreeze task specific parts (lvl2).
        for param in self.encoder[task_id].lvl2encoder.parameters():
            param.requires_grad = unfreeze
        for param in self.decoder[task_id].lvl2decoder.parameters():
            param.requires_grad = unfreeze

    def freeze_lvl1(self, unfreeze=False):
        for param in self.encoder[0].lvl1encoder.parameters():  # shared encoder
            param.requires_grad = unfreeze
        for param in self.encoder[0].attn_layer.parameters(): # shared attention layer
            param.requires_grad = unfreeze
        for param in self.decoder[0].lvl1decoder.parameters():  # shared decoder
            param.requires_grad = unfreeze
        self.encoder[0].lvl1encoder.train()
        self.encoder[0].attn_layer.train()
        self.decoder[0].lvl1decoder.train()
