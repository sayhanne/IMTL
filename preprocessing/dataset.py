import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms


class EffectPredictionDataset(Dataset):
    def __init__(self, task_name, batch_size=500, ext_="pose-scaled", mode="train", y="delta",
                 transform=None, object_id=None):
        X = np.load('{}_data/{}-task-states-{}.npy'.format(mode, task_name, ext_), allow_pickle=True)
        target = np.load('{}_data/{}-task-effects-{}-scaled.npy'.format(mode, task_name, y), allow_pickle=True)
        actions = np.load('{}_data/{}-task-actions.npy'.format(mode, task_name),
                          allow_pickle=True)

        if object_id is None:
            self.actions = actions
            self.X = X
            self.target = target
            self.batch_size = batch_size
            self.transform = transform
        else:
            indices = np.where((actions[:, :6] == object_id).all(axis=1))[0]
            self.actions = np.take(actions, indices, axis=0)
            self.X = np.take(X, indices, axis=0)
            self.target = np.take(target, indices, axis=0)
            self.batch_size = len(indices)
            self.transform = transform

    def __getitem__(self, index):
        if self.transform is not None:
            x = self.transform(self.X[index])
        else:
            x = torch.FloatTensor(self.X[index])
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


def default_transform(size, pad=True, affine=False, mean=None, std=None):
    transform = [transforms.ToPILImage()]
    if pad:
        transform.append(transforms.Pad(padding=size, fill=0, padding_mode='constant'))
    if size:
        transform.append(transforms.Resize(size, interpolation=transforms.InterpolationMode.NEAREST))
    if affine:
        transform.append(
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                fill=int(0.285 * 255)
            )
        )
    transform.append(transforms.ToTensor())
    if mean is not None:
        transform.append(transforms.Normalize([mean], [std]))
    transform = transforms.Compose(transform)
    return transform
