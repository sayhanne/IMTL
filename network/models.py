import os
from collections import OrderedDict

import numpy as np
import torch
from torch import nn

from network.blocks import MLP, Linear
from network.utils.helper import LossUtils, EnergyUtils, TaskSelectionUtils
from copy import deepcopy


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
        self.save_path = config["save"]
        self.task_ids = np.arange(0, config["num_tasks"])
        self.encoder = self.build_encoder(config).to(self.device)
        self.decoder = self.build_decoder(config).to(self.device)
        self.optimizer = torch.optim.Adam(lr=config["learning_rate"],
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

        # Create subdir for plots
        if not os.path.exists(config["save"] + "/plots"):
            os.makedirs(config["save"] + "/plots")

    def build_encoder(self, config):
        raise NotImplementedError

    def build_decoder(self, config):
        raise NotImplementedError

    def loss(self, task_id, sample_state, sample_effect, sample_action):
        state = sample_state.to(self.device)
        action = sample_action.to(self.device)
        effect = sample_effect.to(self.device)

        h = self.encoder[task_id](state)
        h_aug = torch.cat([h, action], dim=-1)
        effect_pred = self.decoder[task_id](h_aug)
        loss = self.criterion(effect_pred, effect)
        return loss

    def one_epoch_optimize(self, task_id, loader):
        self.train()
        running_loss = 0.0
        synaptic_costs = 0.
        for (state, effect, action) in loader:
            self.optimizer.zero_grad()
            loss = self.loss(task_id, state, effect, action)
            loss.backward()
            self.optimizer.step()
            synaptic_costs += self.get_synaptic_cost(task_id)
            running_loss += loss.item()

        avg_loss = running_loss / len(loader)
        avg_cost = synaptic_costs / len(loader)
        self.loss_util.update_task_loss(avg_loss, task_id)
        self.energy_util.update_task_energy(avg_cost, task_id)
        return avg_loss

    def evaluate_epoch(self, task_id, loader, val_=True):
        running_vloss = 0.0
        synaptic_costs = 0.
        self.eval()
        with torch.no_grad():
            for (state, effect, action) in loader:
                loss = self.loss(task_id, state, effect, action)
                running_vloss += loss.item()
                synaptic_costs += self.get_synaptic_cost(task_id)
            avg_vloss = running_vloss / len(loader)
            avg_cost = synaptic_costs / len(loader)
            if val_:
                self.loss_util.update_task_loss(avg_vloss, task_id, is_eval=True)
                self.energy_util.update_task_energy(avg_cost, task_id, is_eval=True)
            else:
                self.loss_util.update_last_loss(avg_vloss, task_id)
                self.energy_util.update_last_energy(avg_cost, task_id)
            return avg_vloss

    def evaluate_all(self, loaders):
        eval_loss = np.zeros(len(self.task_ids))
        for i, t in enumerate(self.task_ids):
            eval_loss[i] = self.evaluate_epoch(t, loaders[t])
        return np.mean(eval_loss)

    # TODO: to be implemented
    # def sum_dense_activations(self, model):
    #     total_activation_sum = 0
    #     for module in model.children():
    #         if isinstance(module, Linear):
    #             if module.activation_out is not None:
    #                 total_activation_sum += module.activation_out
    #         else:
    #             total_activation_sum += self.sum_dense_activations(module)  # Recursive call
    #     return total_activation_sum

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

    def pre_train(self, task_id, tr_loader, val_loader, count=10):
        self.freeze_task(task_id, unfreeze=True)
        for _ in range(count):
            self.one_epoch_optimize(task_id, tr_loader)
            self.evaluate_epoch(task_id, val_loader)
        self.freeze_task(task_id, unfreeze=False)

        # Record pre-train results to the plot
        self.loss_util.train_loss_history_plot[task_id] = deepcopy(self.loss_util.train_loss_history[task_id])

    # TODO: make it more modular
    def train_(self, train_loaders, val_loaders):
        for t in self.task_ids:
            self.pre_train(task_id=t, tr_loader=train_loaders[t], val_loader=val_loaders[t])

        winner_id = self.selection_util.get_winner()
        best_loss = 1e6
        for e in range(self.num_epochs):
            self.freeze_task(task_id=winner_id, unfreeze=True)
            self.one_epoch_optimize(task_id=winner_id, loader=train_loaders[winner_id])
            self.freeze_task(task_id=winner_id)

            if self.model_type == "multitask":
                # Get results again for the other tasks
                for t in self.task_ids:
                    if t != winner_id:
                        self.evaluate_epoch(t, train_loaders[t], val_=False)

            self.loss_util.update_loss_plot()
            avg_vloss = self.evaluate_all(val_loaders)

            if self.selection_type == "lp":
                for t in self.task_ids:
                    self.selectionHelper.calculate_lp(loss=self.lossHelper.train_loss_history[t], index=t)
                self.selectionHelper.save_progress()

            if avg_vloss < best_loss:
                best_loss = avg_vloss
                self.save(self.save_path, "_best")

            # print("Epoch: %d, task: %d, training loss: %.4f, avg task validation loss: %.4f " % (e + 1, winner_id, avg_loss, avg_vloss))
            if e + 1 == self.num_epochs:
                self.save(self.save_path, "_last")

            if e % 200 == 0:
                np.save('{}/plots/train-energy-bar-epoch-{}-seed-{}.npy'.format(self.save_path, e, self.seed),
                        np.asarray(self.energy_util.get_total_energy()))
                np.save('{}/plots/test-energy-bar-epoch-{}-seed-{}.npy'.format(self.save_path, e, self.seed),
                        np.asarray(self.energy_util.get_total_energy(is_eval=True)))

            winner_id = self.selection_util.get_winner()

        # Save final results
        np.save('{}/plots/train-loss-plot-seed-{}.npy'.format(self.save_path, self.seed),
                np.asarray(self.loss_util.train_loss_history_plot))
        np.save('{}/plots/eval-loss-plot-seed-{}.npy'.format(self.save_path, self.seed),
                np.asarray(self.loss_util.eval_loss_history_plot))

    def freeze_all(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        for param in self.encoder.parameters():
            param.requires_grad = True
        for param in self.decoder.parameters():
            param.requires_grad = True

    def freeze_task(self, task_id, unfreeze=False):
        raise NotImplementedError

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

    def freeze_task(self, task_id, unfreeze=False):
        for param in self.encoder[task_id].parameters():
            param.requires_grad = unfreeze
        for param in self.decoder[task_id].parameters():
            param.requires_grad = unfreeze


class MultiTask(EffectPrediction):
    def __init__(self, seed, config):
        super(MultiTask, self).__init__(seed, config)

    def build_encoder(self, config):
        num_tasks = config["num_tasks"]
        task_encoders = nn.ModuleList()
        in_size = max(config["in_size"])
        # Shared encoder
        lvl1encoder = MLP(layer_info=[[in_size] + [config["hidden_dim"]] * config["enc_depth_shared"]],
                          batch_norm=config["batch_norm"])
        for n in range(num_tasks):
            lvl2encoder = MLP(layer_info=[[config["hidden_dim"]] * config["enc_depth_task"] + [config["rep_dim"]]],
                              batch_norm=config["batch_norm"])
            encoder = nn.Sequential(OrderedDict([("lvl1encoder", lvl1encoder),
                                                 ("lvl2encoder", lvl2encoder)]))
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
                layer_info=[[config["hidden_dim"]] * config["dec_depth_shared"] + [config["out_size"][n]]],
                batch_norm=config["batch_norm"])
            decoder = nn.Sequential(OrderedDict([("lvl1decoder", lvl1decoder),
                                                 ("lvl2decoder", lvl2decoder)]))
            task_decoders.append(decoder)
        return task_decoders

    def freeze_task(self, task_id, unfreeze=False):
        # Only freeze/unfreeze task specific parts (lvl2).
        for param in self.encoder[task_id]["lvl2encoder"].parameters():
            param.requires_grad = unfreeze
        for param in self.encoder[task_id]["lvl2decoder"].parameters():
            param.requires_grad = unfreeze

    def freeze_shared(self, unfreeze=False):
        for param in self.encoder[0]["lvl1encoder"].parameters():  # shared encoder
            param.requires_grad = unfreeze
        for param in self.encoder[0]["lvl1decoder"].parameters():  # shared decoder
            param.requires_grad = unfreeze
