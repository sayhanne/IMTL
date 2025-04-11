from copy import deepcopy

import numpy as np


class LossUtils:
    def __init__(self, task_count):
        self.train_loss = [0. for _ in range(task_count)]
        self.eval_loss = [0. for _ in range(task_count)]
        self.train_loss_history = [[] for _ in range(task_count)]
        self.train_loss_history_plot = [[] for _ in range(task_count)]
        self.eval_loss_history_plot = [[] for _ in range(task_count)]

    def update_task_loss(self, loss, index, is_eval=False):
        if not is_eval:
            self.train_loss[index] = loss
            self.train_loss_history[index].append(loss)
        else:
            self.eval_loss[index] = loss
            self.eval_loss_history_plot[index].append(loss)

    def update_last_loss(self, loss, index):
        self.train_loss[index] = loss
        self.train_loss_history[index][-1] = loss

    def update_loss_plot(self, copy=False, task_id=-1):
        if not copy:
            for i, loss in enumerate(self.train_loss):
                self.train_loss_history_plot[i].append(loss)
        else:
            self.train_loss_history_plot[task_id] = deepcopy(self.train_loss_history[task_id])


class EnergyUtils:
    def __init__(self, task_count):
        self.taskCount = task_count
        self.train_energies = [0. for _ in range(task_count)]
        self.eval_energies = [0. for _ in range(task_count)]
        self.train_energy_history = [[] for _ in range(task_count)]
        self.eval_energy_history = [[] for _ in range(task_count)]

    def update_task_energy(self, energy, index, is_eval=False):
        if not is_eval:
            self.train_energies[index] = energy
            self.train_energy_history[index].append(energy)
        else:
            self.eval_energies[index] = energy
            self.eval_energy_history[index].append(energy)

    def update_last_energy(self, energy, index, is_eval=False):
        if not is_eval:
            self.train_energies[index] = energy
            self.train_energy_history[index][-1] = energy
        else:
            self.eval_energies[index] = energy
            self.eval_energy_history[index][-1] = energy

    def get_total_energy(self, is_eval=False):
        total_energy = [0. for _ in range(self.taskCount)]
        if not is_eval:
            for t, en in enumerate(self.train_energy_history):
                total_energy[t] = np.sum(en)

        else:
            for t, en in enumerate(self.eval_energy_history):
                total_energy[t] = np.sum(en)
        return total_energy


class TaskSelectionUtils:
    def __init__(self, task_count, selection, random_seq_path):
        if "lp" in selection:
            self.current_lp = [0. for _ in range(task_count)]
            self.lp_history = [[] for _ in range(task_count)]

        if "e" in selection:
            self.current_ec = [0. for _ in range(task_count)]
            self.ec_history = [[] for _ in range(task_count)]
            self.k = 1

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
        self.current_lp[index] = -slope if slope < 0. else 1e-5

    def calculate_ep(self, energy, index):
        y = energy[-5:]
        self.current_ec[index] = np.sum(y)

    def save_lp(self):
        for i, progress in enumerate(self.current_lp):
            self.lp_history[i].append(progress)

    def save_ec(self):
        for i, energy in enumerate(self.current_ec):
            self.ec_history[i].append(energy)

    def get_winner(self):
        if self.selection == 'rand':
            if self.sequence is not None:
                winner = self.sequence[self.count]  # Equal number of tasks, shuffled sequence (random)
                self.count += 1
            else:
                winner = np.random.randint(0, self.taskCount)  # Pure random
            self.selection_history.append(winner)

        else:  # LP or energy based selection
            winner_index = -1
            if self.selection == "lp":
                winner_index = np.argmax(self.current_lp)
            elif self.selection == "lpe":
                lp_nmpy = np.asarray(self.current_lp)
                lp_normalized = lp_nmpy / np.linalg.norm(lp_nmpy)
                ec_nmpy = np.asarray(self.current_ec)
                ec_normalized = ec_nmpy / np.linalg.norm(ec_nmpy)
                combined = np.exp(self.k * lp_normalized) / ec_normalized
                winner_index = np.argmax(combined)
            selected = np.random.choice(a=[winner_index, -1], p=[1 - self.e, self.e])
            if selected == -1:
                other_tasks = np.setdiff1d(range(self.taskCount), [winner_index])
                winner = np.random.choice(other_tasks)
            else:
                winner = winner_index
            self.selection_history.append(winner)

        return winner
