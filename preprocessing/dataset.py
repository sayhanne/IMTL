import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class LocationPredictionDataset(Dataset):
    def __init__(self, task_name, batch_size=500, mode="train"):  # mode -> train, val, test
        X = np.load('{}_data/{}-task-states-pose-scaled.npy'.format(mode, task_name), allow_pickle=True)
        target = np.load('{}_data/{}-task-effects-pose-scaled.npy'.format(mode, task_name), allow_pickle=True)
        actions = np.load('{}_data/{}-task-actions.npy'.format(mode, task_name),
                          allow_pickle=True)

        self.actions = actions
        self.X = X
        self.target = target
        self.batch_size = batch_size

    def __getitem__(self, index):
        x = torch.FloatTensor(self.X[index])  # start position
        y = torch.FloatTensor(self.target[index])  # end position
        a = torch.FloatTensor(self.actions[index])  # action

        return x, y, a

    def __len__(self):
        return len(self.target)

    def load_data(self, shuffle=False):
        loader = DataLoader(
            self,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )

        return loader
