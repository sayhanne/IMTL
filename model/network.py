import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader

from model.backbone import BackBoneNet
from model.layers import DenseLayer
from model.subnet import SubNet


class MTL(nn.Module):
    def __init__(self, use_cuda=True, shared_backbone=True):
        super(MTL, self).__init__()
        # Model settings
        self.network = nn.ModuleList()
        self.shared_backbone = shared_backbone
        # cuda settings
        use_cuda = use_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        # Training settings
        self.optimizer = None
        self.criterion = nn.L1Loss()

    def create_network(self, num_hid, num_tasks, input_dims, hidden_dim, output_dims):
        backbone = BackBoneNet(hidden_dim=hidden_dim, num_layers=num_hid, activation=nn.ReLU(), device=self.device)

        for t in range(num_tasks):
            input_layer = DenseLayer(inSize=input_dims[t], outSize=hidden_dim, activation=nn.ReLU(),
                                     device=self.device)
            if not self.shared_backbone:  # if not sharing backbone, create independent backbone
                backbone = BackBoneNet(hidden_dim=hidden_dim, num_layers=num_hid, activation=nn.ReLU(),
                                       device=self.device)
            output_layers = SubNet(hidden_dim=hidden_dim, out_dim=output_dims[t], num_layers=num_hid // 2,
                                   activation=nn.ReLU(), device=self.device)
            task_net = nn.Sequential(input_layer, backbone, output_layers)
            self.network.append(task_net)

    def set_optimizer(self):
        self.optimizer = optim.AdamW(params=self.parameters(), lr=1e-4, weight_decay=1e-3, amsgrad=True)

    def freeze_all(self):
        for param in self.network.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        for param in self.network.parameters():
            param.requires_grad = True

    def freeze_task(self, task_id, unfreeze=False):
        self.network[task_id][0].freeze(unfreeze)  # DenseLayer (projection)
        self.network[task_id][2].freeze(unfreeze)  # SubNet
        if not self.shared_backbone:  # independent BackBoneNet
            self.network[task_id][1].freeze(unfreeze)

    def freeze_backbone(self, unfreeze=False):
        self.network[0][1].freeze(unfreeze)  # task id does not matter. all tasks share same backbone

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
        action_X = torch.hstack((action, X))
        y = self.network[task_id](action_X)
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
                batch_energy += self.sum_dense_activations(self.network[task_id]) / 1000    # as in kW
        avg_loss = running_loss / len(data_loader)
        avg_energy = batch_energy / len(data_loader)

        return avg_loss, avg_energy
