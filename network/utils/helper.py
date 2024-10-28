import math
import numpy as np


class LossUtils:
    def __init__(self, task_count):
        self.train_losses = [0. for _ in range(task_count)]
        self.test_losses = [0. for _ in range(task_count)]
        self.train_loss_history = [[] for _ in range(task_count)]
        self.train_loss_history_plot = [[] for _ in range(task_count)]
        self.test_loss_history_plot = [[] for _ in range(task_count)]

    def update_task_loss(self, index, is_eval=False):
        if not is_eval:
            self.train_loss_history[index].append(self.train_losses[index])
        else:
            self.train_loss_history[index][-1] = self.train_losses[index]

    def update_task_loss_plots(self):
        for i, loss in enumerate(self.train_losses):
            self.train_loss_history_plot[i].append(loss)

    def update_test_loss(self, index):
        self.test_loss_history_plot[index].append(self.test_losses[index])


class EnergyUtils:
    def __init__(self, task_count):
        self.taskCount = task_count
        self.train_energies = [0. for _ in range(task_count)]
        self.test_energies = [0. for _ in range(task_count)]
        self.train_energy_history = [[] for _ in range(task_count)]
        self.test_energy_history = [[] for _ in range(task_count)]
        # self.train_energy_history_plot = [[] for _ in range(task_count)]

    def update_task_energy(self, index, is_eval=False):
        if not is_eval:
            self.train_energy_history[index].append(self.train_energies[index])
        else:
            self.train_energy_history[index][-1] = self.train_energies[index]

    def update_test_energy(self, index):
        self.test_energy_history[index].append(self.test_energies[index])

    def get_total_energy(self, is_test=False):
        total_energy = [0. for _ in range(self.taskCount)]
        if not is_test:
            for t, en in enumerate(self.train_energy_history):
                total_energy[t] = np.sum(en)

        else:
            for t, en in enumerate(self.test_energy_history):
                total_energy[t] = np.sum(en)
        return total_energy


class TaskSelectionUtils:
    def __init__(self, task_count, selection, random_seq_path):
        self.current_lp = [0. for _ in range(task_count)]
        self.lp_history = [[] for _ in range(task_count)]

        self.selection_history = []
        self.selection = selection
        self.taskCount = task_count
        self.e = 0.1
        self.count = 0
        if self.selection == 'rand':
            if random_seq_path is not None:
                self.sequence = np.load(random_seq_path)
            else:
                self.sequence = None

    def calculate_lp(self, loss, index):
        y = loss[-5:]
        x = range(1, 6)
        slope = np.polyfit(x, y, deg=1)[0]
        if slope < 0.:
            self.current_lp[index] = math.fabs(slope)

    def save_progress(self):
        for i, progress in enumerate(self.current_lp):
            self.lp_history[i].append(progress)

    def get_winner(self):
        if self.selection == 'rand':
            if self.sequence is not None:
                winner = self.sequence[self.count]  # Equal number of tasks, shuffled sequence (random)
                self.count += 1
            else:
                winner = np.random.randint(0, self.taskCount)  # Pure random
            self.selection_history.append(winner)

        else:  # LP based selection
            winner_index = np.argsort(self.current_lp)[::-1][0]  # Highest lp
            selected = np.random.choice(a=[winner_index, -1], p=[1 - self.e, self.e])
            if selected == -1:
                other_tasks = np.setdiff1d(range(self.taskCount), [winner_index])
                winner = np.random.choice(other_tasks)
            else:
                winner = winner_index
            self.selection_history.append(winner)

        return winner
