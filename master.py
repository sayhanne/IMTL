import numpy as np
import torch
from torch import nn

from model.network import MTL
from model.utils.helper import LossUtils, TaskSelectionUtils
from preprocessing.dataset import LocationPredictionDataset


class Master:
    def __init__(self, model_name, use_cuda=True, mb_size=500, epochs=600,
                 selection='rand', random_seq_path=None, save_path='results/'):
        # Training settings
        self.taskIDs = None  # will be initialized later
        self.data_loaders = []  # The data loaders will be created while reading data from path
        self.validation_data_loaders = []  # The data loaders will be created while reading data from path
        self.mb_size = mb_size
        self.epochs_done = 0
        self.epochs = epochs
        self.selection = selection
        self.model = MTL(use_cuda=use_cuda)
        self.model_name = model_name
        self.random_seq_path = random_seq_path

        # Test settings
        self.test_loaders = []  # The data loaders will be created while reading data from path

        # Loss helper
        self.lossHelper = None  # Set according to task count!
        # Task selection helper
        self.selectionHelper = None  # Set according to task count!

        # Save path
        self.save_path = save_path

    def init_model(self, task_names, is_train=True):
        input_dims = []
        output_dims = []
        self.taskIDs = np.arange(start=0, stop=len(task_names))
        for task_name in task_names:
            if task_name == 'push' or task_name == 'hit':
                input_dims.append(10)  # 7 + 3
                output_dims.append(3)
            elif task_name == 'stack':
                input_dims.append(16)  # 10 + 6
                output_dims.append(6)

            if is_train:
                # Train set
                task_dataset = LocationPredictionDataset(task_name=task_name, batch_size=self.mb_size, mode="train")
                task_loader = task_dataset.load_data(shuffle=True)
                self.data_loaders.append(task_loader)

                # Test set
                test_dataset = LocationPredictionDataset(task_name=task_name, batch_size=self.mb_size, mode="test")
                test_task_loader = test_dataset.load_data()
                self.test_loaders.append(test_task_loader)
            else:
                pass
                # TODO: later
                # test_dataset = LocationPredictionDataset(task_name=task_name, batch_size=self.mb_size,
                #                                          mode="test")
                # test_task_loader = test_dataset.load_data()
                # self.test_loaders.append(test_task_loader)

        self.model.create_network(num_layers=4, input_dims=input_dims, hidden_dim=6, output_dims=output_dims)
        self.model.freeze_all()
        self.model.set_optimizer()

        # Utils
        if is_train:
            self.lossHelper = LossUtils(task_count=len(task_names))
            self.selectionHelper = TaskSelectionUtils(task_count=len(task_names), selection=self.selection,
                                                      random_seq_path=self.random_seq_path)

    def train_task(self, task_id, initial=False):
        if not initial:
            self.model.freeze_task(task_id=task_id, unfreeze=True)
        error = self.model.train_mb(task_id=task_id, data_loader=self.data_loaders[task_id])
        print("Task {}, error {}".format(task_id, error))
        if not initial:
            self.model.freeze_task(task_id=task_id)
        self.lossHelper.train_losses[task_id] = error
        self.lossHelper.update_task_loss(index=task_id)

    def test_task(self, task_id):
        test_loss = self.model.evaluate_mb(task_id=task_id, data_loader=self.test_loaders[task_id])
        self.lossHelper.test_losses[task_id] = test_loss
        self.lossHelper.update_test_loss(task_id)

    def test_tasks(self):
        for t in self.taskIDs:
            self.test_task(task_id=t)

    def initial_run(self, count=10):

        print('--INITIAL RUN START---')
        for t in self.taskIDs:  # Train tasks 'count' times to bootstrap lp calculation
            self.model.freeze_task(task_id=t, unfreeze=True)
            for _ in range(count):
                self.train_task(t, initial=True)
                self.test_task(t)
            self.model.freeze_task(task_id=t)

        if self.selection == 'lp':
            for t in self.taskIDs:
                self.selectionHelper.calculate_lp(loss=self.lossHelper.train_loss_history[t], index=t)

    def run(self):
        self.model.freeze_backbone(unfreeze=True)
        winner = self.selectionHelper.get_winner()
        self.epochs_done = 0

        for _ in range(self.epochs):

            # Train task
            self.train_task(task_id=winner)
            self.lossHelper.update_task_loss_plots()

            # Save results
            if self.selection == 'lp':
                for t in self.taskIDs:
                    self.selectionHelper.calculate_lp(loss=self.lossHelper.train_loss_history[t], index=t)
                self.selectionHelper.save_progress()

            self.test_tasks()
            self.epochs_done += 1
            if self.epochs_done < self.epochs:
                winner = self.selectionHelper.get_winner()

            # if self.epochs_done % 200 == 0:
            #     torch.save(self.model.state_dict(), 'saved_models/{}-epoch={}-state-dict.pt'.format(self.model_name,
            #                                                                                         self.epochs_done))

        # Save final results
        np.save('{}/train-loss-plot-{}.npy'.format(self.save_path, self.model_name),
                np.asarray(self.lossHelper.train_loss_history_plot))
        np.save('{}/test-loss-plot-{}.npy'.format(self.save_path, self.model_name),
                np.asarray(self.lossHelper.test_loss_history_plot))


if __name__ == '__main__':
    seed = 3563646
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    task_names = ['push', 'hit', 'stack']
    master = Master(model_name='lp', selection='lp')
    master.init_model(task_names)
    master.initial_run()
    master.run()
