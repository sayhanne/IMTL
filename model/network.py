import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader

from model.backbone import BackBoneNet
from model.layers import DenseLayer
from model.subnet import SubNet


class MTL(nn.Module):
    def __init__(self, use_cuda=True):
        super().__init__()
        # Model settings
        self.network = nn.ModuleList()
        # cuda settings
        use_cuda = use_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        # Training settings
        self.optimizer = None
        self.criterion = nn.L1Loss()

    def create_network(self, num_layers, input_dims, hidden_dim, output_dims):
        input_layers = nn.ModuleList([DenseLayer(inSize=i, outSize=hidden_dim,  # projection layers
                                                 activation=nn.ReLU(), device=self.device)
                                      for i in input_dims])
        self.network.append(input_layers)
        self.network.append(BackBoneNet(hidden_dim=hidden_dim, num_layers=num_layers, device=self.device))
        task_heads = nn.ModuleList([SubNet(hidden_dim=hidden_dim, out_dim=o,  # Task specific small networks.
                                           num_layers=num_layers // 2, device=self.device)
                                    for o in output_dims])
        self.network.append(task_heads)

    def set_optimizer(self):
        self.optimizer = optim.AdamW(params=self.parameters(), lr=1e-4, weight_decay=1e-3, amsgrad=True)

    def freeze_all(self):
        for param in self.network.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        for param in self.network.parameters():
            param.requires_grad = True

    def freeze_task(self, task_id, unfreeze=False):
        self.network[0][task_id].freeze(unfreeze)   # projection layer
        self.network[2][task_id].freeze(unfreeze)      # task subnet

    def freeze_backbone(self, unfreeze=False):
        self.network[1].freeze(unfreeze)        # backbone network

    def forward(self, task_id, action, X):
        action_X = torch.hstack((action, X))
        projected_x = self.network[0][task_id](action_X)    # input to backbone projection
        shared_rep = self.network[1](projected_x)       # backbone
        y = self.network[2][task_id](shared_rep)      # task subnet
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
        for (X, target, action) in data_loader:
            self.optimizer.zero_grad()
            batch_loss = self.forward_mb(task_id=task_id, y=target, action=action, X=X)
            batch_loss.backward()
            self.optimizer.step()
        eval_error = self.evaluate_mb(task_id=task_id, data_loader=data_loader)  # evaluate on training data
        return eval_error

    def evaluate_mb(self, task_id, data_loader: DataLoader):
        self.eval()
        running_loss = 0.0
        with torch.no_grad():
            for (X, target, action) in data_loader:
                batch_loss = self.forward_mb(task_id=task_id, y=target, action=action, X=X)
                running_loss += batch_loss.item()
        avg_loss = running_loss / len(data_loader)

        return avg_loss
