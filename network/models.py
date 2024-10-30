import os
from collections import OrderedDict

import torch
from torch import nn

from network.blocks import MLP
from utils.helper import LossUtils, EnergyUtils, TaskSelectionUtils


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
    def __init__(self, config):
        super(EffectPrediction, self).__init__()
        self.device = torch.device(config["device"])
        self.criterion = torch.nn.L1Loss()
        self.iteration = 0
        self.save_path = config["save"]
        self.task_ids = range(config["num_tasks"])
        self.encoder = self.build_encoder(config).to(self.device)
        self.decoder = self.build_decoder(config).to(self.device)
        self.optimizer = torch.optim.Adam(lr=config["learning_rate"],
                                          params=[
                                              {"params": self.encoder.parameters()},
                                              {"params": self.decoder.parameters()}],
                                          amsgrad=True)
        self.loss_util = LossUtils(config["num_tasks"])
        self.energy_util = EnergyUtils(config["num_tasks"])
        self.selection_util = TaskSelectionUtils(config["num_tasks"], config["selection"],
                                                 config["task_sequence"])

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
        for (state, effect, action) in loader:
            self.optimizer.zero_grad()
            loss = self.loss(task_id, state, effect, action)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            self.iteration += 1
        return running_loss / len(loader)

    def evaluate_epoch(self, task_id, loader):
        running_vloss = 0.0
        self.eval()
        with torch.no_grad():
            for (state, effect, action) in loader:
                loss = self.loss(task_id, state, effect, action)
                running_vloss += loss.item()
            return running_vloss / len(loader)

    def pre_train(self, task_id, tr_loader, val_loader, count=10):
        self.freeze_task(task_id, unfreeze=True)
        for _ in range(count):
            # TODO: loss, selection, energy
            train_loss = self.one_pass_optimize(task_id, tr_loader)
            val_loss = self.evaluate_epoch(task_id, val_loader)
        self.freeze_task(task_id, unfreeze=False)

    def train_(self, epoch, train_loaders, val_loaders):
        for t in self.task_ids:
            self.pre_train(task_id=t, tr_loader=train_loaders[t], val_loader=val_loaders[t])

        winner_id = self.selection_util.get_winner()        # TODO: update
        best_loss = [1e6 for _ in range(len(self.task_ids))]
        for e in range(epoch):
            avg_loss = self.one_epoch_optimize(task_id=winner_id, loader=train_loaders[winner_id])
            avg_vloss = self.evaluate_epoch(task_id=winner_id, loader=val_loaders[winner_id])
            # TODO: loss, selection, energy
            if avg_vloss < best_loss[winner_id]:
                best_loss[winner_id] = avg_vloss
                self.save(self.save_path, "_best")
            print("Epoch: %d, training loss: %.4f, validation loss: %.4f " % (e + 1, avg_loss, avg_vloss))
            self.save(self.save_path, "_last")

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

        encoder_dict = torch.load(os.path.join(path, "encoder" + ext + ".ckpt"))
        decoder_dict = torch.load(os.path.join(path, "decoder" + ext + ".ckpt"))
        encoder.load_state_dict(encoder_dict)
        decoder.load_state_dict(decoder_dict)

    def save(self, path, ext):
        encoder = self.encoder
        decoder = self.decoder

        encoder_dict = encoder.eval().cpu().state_dict()
        decoder_dict = decoder.eval().cpu().state_dict()
        torch.save(encoder_dict, os.path.join(path, "encoder" + ext + ".ckpt"))
        torch.save(decoder_dict, os.path.join(path, "decoder" + ext + ".ckpt"))
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
    def __init__(self, config):
        super(SingleTask, self).__init__(config)

    def build_encoder(self, config):
        num_tasks = config["num_tasks"]
        task_encoders = nn.ModuleList()
        for n in range(num_tasks):
            lvl1encoder = MLP(layer_info=[[config["in_size"][n]] + [config["hidden_dim"]] * config["enc_depth_shared"]],
                              batch_norm=config["batch_norm"])
            lvl2encoder = MLP(layer_info=[[config["hidden_dim"]] * config["enc_depth_task"] + [config["rep_dim"]]],
                              batch_norm=config["batch_norm"])
            encoder = nn.Sequential(OrderedDict([("lvl1encoder", lvl1encoder),
                                                 ("lvl2encoder", lvl2encoder)]))
            task_encoders.append(encoder)
        return task_encoders

    def build_decoder(self, config):
        num_tasks = config["num_tasks"]
        task_decoders = nn.ModuleList()
        for n in range(num_tasks):
            lvl1decoder = MLP(layer_info=[config["rep_dim"] + config["act_dim"]] + [config["hidden_dim"]] * config[
                "dec_depth_shared"],
                              batch_norm=config["batch_norm"])
            lvl2decoder = MLP(
                layer_info=[[config["hidden_dim"]] * config["dec_depth_shared"] + [config["out_size"][n]]],
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
    def __init__(self, config):
        super(MultiTask, self).__init__(config)

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
        lvl1decoder = MLP(layer_info=[config["rep_dim"] + config["act_dim"]] + [config["hidden_dim"]] * config[
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
