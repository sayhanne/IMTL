import os
from argparse import ArgumentParser

import numpy as np
import torch
from torch import nn

from model.network import MTL
from model.utils.helper import LossUtils, TaskSelectionUtils, EnergyUtils
from preprocessing.dataset import LocationPredictionDataset


class Master:
    def __init__(self, model_name, use_cuda=True, mb_size=500, epochs=600, shared_backbone=True,
                 selection='rand', random_seq_path=None, save_path='results/'):
        # Training settings
        self.taskIDs = None  # will be initialized later
        self.data_loaders = []  # The data loaders will be created while reading data from path
        self.validation_data_loaders = []  # The data loaders will be created while reading data from path
        self.mb_size = mb_size
        self.epochs_done = 0
        self.epochs = epochs
        self.selection = selection
        self.shared_backbone = shared_backbone
        self.model = MTL(use_cuda=use_cuda, shared_backbone=shared_backbone)
        self.model_name = model_name
        self.random_seq_path = random_seq_path

        # Test settings
        self.test_loaders = []  # The data loaders will be created while reading data from path

        # Loss helper
        self.lossHelper = None  # Set according to task count!
        # Energy helper
        self.energyHelper = None  # Set according to task count!
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
                input_dims.append(14)  # 12 + 2
                output_dims.append(12)
            elif task_name == 'stack':
                input_dims.append(26)  # 24 + 2
                output_dims.append(24)

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

        self.model.create_network(num_hid=4, num_tasks=len(task_names), input_dims=input_dims, hidden_dim=4,
                                  output_dims=output_dims)
        self.model.freeze_all()
        self.model.set_optimizer()

        # Utils
        if is_train:
            self.lossHelper = LossUtils(task_count=len(task_names))
            self.energyHelper = EnergyUtils(task_count=len(self.taskIDs))
            self.selectionHelper = TaskSelectionUtils(task_count=len(task_names), selection=self.selection,
                                                      random_seq_path=self.random_seq_path)

    def train_task(self, task_id):
        error, energy = self.model.train_mb(task_id=task_id, data_loader=self.data_loaders[task_id])
        # print("Task {}, error {}".format(task_id, error))
        self.lossHelper.train_losses[task_id] = error
        self.lossHelper.update_task_loss(index=task_id)
        self.energyHelper.train_energies[task_id] = energy
        self.energyHelper.update_task_energy(index=task_id)

    def test_task(self, task_id):
        test_loss, test_energy = self.model.evaluate_mb(task_id=task_id, data_loader=self.test_loaders[task_id])
        self.lossHelper.test_losses[task_id] = test_loss
        self.lossHelper.update_test_loss(task_id)
        self.energyHelper.test_energies[task_id] = test_energy
        self.energyHelper.update_test_energy(task_id)

    def test_tasks(self):
        for t in self.taskIDs:
            self.test_task(task_id=t)

    def initial_run(self, count=10):

        # print('--INITIAL RUN START---')
        for t in self.taskIDs:  # Train tasks 'count' times to bootstrap learning
            self.model.freeze_task(task_id=t, unfreeze=True)
            for _ in range(count):
                self.train_task(t)
                self.test_task(t)
            self.model.freeze_task(task_id=t)

        if self.selection == 'lp':
            for t in self.taskIDs:
                self.selectionHelper.calculate_lp(loss=self.lossHelper.train_loss_history[t], index=t)

    def run(self):
        if self.shared_backbone:
            self.model.freeze_backbone(unfreeze=True)
        winner = self.selectionHelper.get_winner()
        self.epochs_done = 0

        for _ in range(self.epochs):

            # Train task
            self.model.freeze_task(task_id=winner, unfreeze=True)
            self.train_task(task_id=winner)
            self.model.freeze_task(task_id=winner)
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

            if self.epochs_done % 200 == 0:
                np.save('{}/train-energy-bar-{}-epoch-{}.npy'.format(self.save_path, self.model_name, self.epochs_done),
                        np.asarray(self.energyHelper.get_total_energy()))
                np.save('{}/test-energy-bar-{}-epoch-{}.npy'.format(self.save_path, self.model_name, self.epochs_done),
                        np.asarray(self.energyHelper.get_total_energy(is_test=True)))
            #     torch.save(self.model.state_dict(), 'saved_models/{}-epoch={}-state-dict.pt'.format(self.model_name,
            #                                                                                         self.epochs_done))

        # Save final results
        np.save('{}/train-loss-plot-{}.npy'.format(self.save_path, self.model_name),
                np.asarray(self.lossHelper.train_loss_history_plot))
        np.save('{}/test-loss-plot-{}.npy'.format(self.save_path, self.model_name),
                np.asarray(self.lossHelper.test_loss_history_plot))


class Runner:
    def __init__(self, lock, seed, name, selection, shared_backbone, epoch, seq_path, save_path):
        lock.acquire()
        try:
            self.seed = int(seed)
            self.name = name
            self.selection = selection
            self.shared_backbone = True if int(shared_backbone) else False
            self.num_epochs = int(epoch)
            self.sequence = seq_path
            self.save_path = save_path

            torch.manual_seed(int(seed))
            np.random.seed(int(seed))

            print('Module name:', __name__)
            print('parent process:', os.getppid())
            print('process id:', os.getpid())
            print('Seed:', self.seed)
            print('Name:', self.name)
            print('Selection', self.selection)
            print('Sharing backbone', self.shared_backbone)
            print('Sequence:', self.sequence)
            print("-------")
        finally:
            lock.release()
        # print(seed)

        # Training
        tasks = ['push', 'stack', 'hit']
        parser = ArgumentParser(prog='IMTL',
                                description='Interleaved Multi-Task Learning with Energy Modulated Learning Progress',
                                epilog='OZU COG-ROB LAB')

        parser.add_argument('--use-cuda', action='store_true', default=True,
                            help='allow the use of CUDA (default: True)')

        parser.add_argument('-ep', '--train-epochs', type=int, default=self.num_epochs,
                            help='total number of epochs to train all tasks(default: 600)')

        parser.add_argument('-bs', '--train-batch-size', type=int, default=500,
                            help='batch size in training (default: 500)')

        parser.add_argument('-ts', '--task-selection', default=self.selection, choices=['rand', 'lp', 'lpe'],
                            help='Task selection {rand: Random selection, lp: Learning progress '
                                 'based selection, lpe: Energy modulated lp selection} '
                                 '(default: "rand")')
        parser.add_argument('-sbb', '--shared-backbone', default=self.shared_backbone, choices=[True, False],
                            help='Backbone setting {True: Shared, False: Independent} '
                                 '(default: "True")')

        parser.add_argument('-rs', '--rand-seq-path', default=self.sequence,
                            help='Random task sequence path (default: None)')

        parser.add_argument('-sp', '--save-path', default=self.save_path,
                            help='Save results path (default: "results/")')

        args = parser.parse_args()

        master = Master(model_name='{}-seed-{}'.format(self.name, self.seed),
                        use_cuda=args.use_cuda, mb_size=args.train_batch_size,
                        epochs=args.train_epochs, selection=args.task_selection,
                        shared_backbone=args.shared_backbone,
                        random_seq_path=args.rand_seq_path, save_path=args.save_path)

        master.init_model(tasks)
        master.initial_run()
        master.run()
