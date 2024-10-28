import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class LocationPredictionDataset(Dataset):
    def __init__(self, task_name, batch_size=500, mode="train", target_obj=-1, moving_obj=-1):  # mode -> train, val, test
        X = np.load('{}_data/{}-task-states-pose-scaled.npy'.format(mode, task_name), allow_pickle=True)
        target = np.load('{}_data/{}-task-effects-pose.npy'.format(mode, task_name), allow_pickle=True)
        actions = np.load('{}_data/{}-task-actions.npy'.format(mode, task_name),
                          allow_pickle=True)

        # obj_encoded = {
        #     1: [1, 0, 0, 0, 0],  # sphere
        #     2: [0, 1, 0, 0, 0],  # box
        #     3: [0, 0, 1, 0, 0],  # cylinder
        #     4: [0, 0, 0, 1, 0],  # capsule
        #     5: [0, 0, 0, 0, 1]  # prism
        # }
        #
        # if target_obj != -1:
        #     if task_name != "stack":
        #         one_hot = obj_encoded[target_obj]
        #         indices = np.where(np.all(actions[:, 2:] == one_hot, axis=1))[0]
        #     else:
        #         one_hot_target = obj_encoded[target_obj]
        #         # one_hot_moving = obj_encoded[moving_obj]
        #         # one_hot = np.hstack((one_hot_target, one_hot_moving))
        #         indices = np.where(np.all(actions[:, :5] == one_hot_target, axis=1))[0]
        #     self.actions = actions[indices]
        #     self.X = X[indices]
        #     self.target = target[indices]
        #     self.batch_size = len(indices)

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
