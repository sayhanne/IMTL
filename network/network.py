import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader

from network.blocks import DenseLayer, MLP, MultiTaskAttention

TASKS = ['push', 'hit', 'stack']
INPUT_DIM_SINGLE = 12
INPUT_DIM_STACK = 24
UNIFIED_INPUT_DIM = INPUT_DIM_STACK  # 24 dimensions
ACTION_DIM = 2
HIDDEN_DIM = 32
LATENT_DIM = 16
SHARED_ENC_DIM = 64


def pad_input(input_tensor, target_dim):
    padding = target_dim - input_tensor.size(1)
    if padding > 0:
        padded_input = F.pad(input_tensor, (0, padding))
    else:
        padded_input = input_tensor
    return padded_input


class MTL(nn.Module):
    def __init__(self, use_cuda=True, is_sharing=True):
        super(MTL, self).__init__()
        # Model settings
        self.network = nn.ModuleList()
        self.is_sharing = is_sharing
        # cuda settings
        use_cuda = use_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        # Training settings
        self.optimizer = None
        self.criterion = nn.L1Loss()

    def create_network(self):
        backbone_encoder, attention_net, action_layer, backbone_decoder = self.create_network_parts()

        for task in TASKS:
            if not self.is_sharing:
                backbone_encoder, attention_net, action_layer, backbone_decoder = self.create_network_parts()
            task_encoder = MLP(input_dim=SHARED_ENC_DIM, hidden_dim=HIDDEN_DIM, out_dim=LATENT_DIM, num_layers=2,
                               activation=nn.ReLU(), device=self.device, is_shared=False)
            output_dim = INPUT_DIM_SINGLE if task != 'stack' else INPUT_DIM_STACK
            task_decoder = MLP(input_dim=SHARED_ENC_DIM, hidden_dim=HIDDEN_DIM, out_dim=output_dim, num_layers=2,
                               activation=nn.ReLU(), device=self.device, is_shared=False)
            task_network = nn.ModuleList([backbone_encoder, task_encoder, attention_net,
                                          action_layer, backbone_decoder, task_decoder])
            self.network.append(task_network)

    def create_network_parts(self):
        encoder = MLP(input_dim=UNIFIED_INPUT_DIM, hidden_dim=HIDDEN_DIM, out_dim=SHARED_ENC_DIM, num_layers=4,
                      activation=nn.ReLU(), device=self.device, is_shared=self.is_sharing)
        attention_net = MultiTaskAttention(latent_dim=LATENT_DIM, is_shared=self.is_sharing).to(self.device)
        action_concat = DenseLayer(inSize=LATENT_DIM + ACTION_DIM, outSize=LATENT_DIM,
                                   activation=nn.ReLU(), device=self.device, is_shared=self.is_sharing)
        decoder = MLP(input_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM, out_dim=SHARED_ENC_DIM, num_layers=4,
                      activation=nn.ReLU(), device=self.device, is_shared=self.is_sharing)
        return encoder, attention_net, action_concat, decoder

    def set_optimizer(self):
        self.optimizer = optim.AdamW(params=self.parameters(), lr=1e-4, weight_decay=1e-3, amsgrad=True)

    def freeze_all(self):
        for param in self.network.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        for param in self.network.parameters():
            param.requires_grad = True

    def freeze_task(self, task_id, unfreeze=False):
        for module in self.network[task_id]:
            if not module.is_shared:
                module.freeze(unfreeze)

    def freeze_shared_(self, unfreeze=False):
        for module in self.network[0]:      # task id does not matter
            if module.is_shared:
                module.freeze(unfreeze)

    def sum_dense_activations(self, model):
        total_activation_sum = 0
        for module in model.children():
            if isinstance(module, DenseLayer):
                if module.activation_out is not None:
                    total_activation_sum += module.activation_out
            else:
                total_activation_sum += self.sum_dense_activations(module)  # Recursive call
        return total_activation_sum

    def get_synaptic_cost(self, task_id):
        synaptic_transmission_cost = 0
        for param in self.network[task_id].parameters():
            if param.requires_grad:
                weight = param.data.detach().cpu().numpy()
                gradient = param.grad.detach().cpu().numpy()
                # Element-wise multiplication of weights and gradients
                cost = np.sum(np.abs(weight) * np.abs(gradient))
                synaptic_transmission_cost += cost
        return synaptic_transmission_cost

    def forward(self, task_id, action, X):
        x = pad_input(X, UNIFIED_INPUT_DIM)
        enc1_out = self.network[task_id][0](x)
        enc2_out = self.network[task_id][1](enc1_out)
        att_out = self.network[task_id][2](enc2_out)

        action_x = torch.hstack((action, att_out))
        action_out = self.network[task_id][3](action_x)
        dec1_out = self.network[task_id][4](action_out)
        y = self.network[task_id][5](dec1_out)

        return y

    def forward_mb(self, task_id, y: Tensor, action: Tensor, X: Tensor):
        effect = y.to(self.device)
        action = action.to(self.device)
        x = X.to(self.device)
        prediction = self(task_id, action=action, X=x)
        loss = self.criterion(prediction, effect)
        return loss

    def train_mb(self, task_id, data_loader: DataLoader):
        self.train()
        synaptic_costs = 0.
        for (X, target, action) in data_loader:
            self.optimizer.zero_grad()
            batch_loss = self.forward_mb(task_id=task_id, y=target, action=action, X=X)
            batch_loss.backward()
            synaptic_costs += self.get_synaptic_cost(task_id)
            self.optimizer.step()
        avg_synaptic_costs = synaptic_costs / len(data_loader)
        eval_error, eval_activation = self.evaluate_mb(task_id=task_id,
                                                       data_loader=data_loader)  # evaluate on training data
        eval_energy = eval_activation + avg_synaptic_costs
        return eval_error, eval_energy

    def evaluate_mb(self, task_id, data_loader: DataLoader):
        self.eval()
        running_loss = 0.0
        batch_energy = 0.
        with torch.no_grad():
            for (X, target, action) in data_loader:
                batch_loss = self.forward_mb(task_id=task_id, y=target, action=action, X=X)
                running_loss += batch_loss.item()
                batch_energy += self.sum_dense_activations(self.network[task_id]) / 1000  # as in kW
        avg_loss = running_loss / len(data_loader)
        avg_energy = batch_energy / len(data_loader)

        return avg_loss, avg_energy
