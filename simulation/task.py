import os

import numpy as np

from environment import PushEnv, StackEnv, HitEnv, CollisionEnv
import math

REAL_TIME = True


class TableTopTask:
    def __init__(self, data_path, N, task_name):
        in_dim = 3
        """ total 6 objs => 6 dim for one-hot obj id """
        action_dim = 8     # [sin(x), cos(x), one-hot obj-id]
        out_dim = 3
        self.upper_lim = 180.
        if task_name == 'push':
            self.env = PushEnv(gui=1)
            self.obj_type = self.env.initialize()

        elif task_name == 'stack':
            self.env = StackEnv(gui=1)
            self.obj_type = self.env.initialize()
            in_dim = 6
            out_dim = 6
            action_dim = 12     # [one-hot 1st obj-id, one-hot 2nd obj-id]

        elif task_name == 'hit':
            self.env = HitEnv(gui=1)
            self.obj_type = self.env.initialize()

        elif task_name == 'collision':
            self.env = CollisionEnv(gui=1)
            self.obj_type = self.env.initialize()
            in_dim = 6
            out_dim = 6
            action_dim = 14  # [sin(x), cos(x), one-hot 1st obj-id, one-hot 2nd obj-id]

        self.cursor = 0
        self.task_name = task_name

        # start collecting data set
        self.path = data_path
        self.state_img = np.zeros((int(N), 256, 256), dtype=np.uint8)  # state (256 x 256 depth image)
        self.effect_img = np.zeros((int(N), 256, 256), dtype=np.uint8)  # effect (256 x 256 depth image)

        self.state_pose = np.zeros((int(N), in_dim))  # start location (Cartesian location)
        self.effect_pose = np.zeros((int(N), out_dim))  # effect location (Cartesian location)

        self.action_data = np.zeros((int(N), action_dim))  # action (sin, cos) and obj type(s)

        self.num_samples = int(N)
        self.index = 0
        self.collect()

    def run_episode(self):
        action_choice = np.random.uniform(0., self.upper_lim, size=1)
        state, effect = self.env.step(action=action_choice, sleep=REAL_TIME)

        self.state_img[self.index] = list(state)[0]
        self.state_pose[self.index] = list(state)[1]
        self.effect_img[self.index] = list(effect)[0]
        self.effect_pose[self.index] = list(effect)[1]

        if self.task_name != "stack":
            action = np.hstack(([math.sin(math.radians(action_choice)), math.cos(math.radians(action_choice))],
                                self.obj_type.copy()))
        else:
            action = self.obj_type.copy()
        self.action_data[self.index] = action
        self.index += 1
        if self.index == self.num_samples:
            return

        self.changeTypes()

    def changeTypes(self):
        if self.task_name == 'push' or self.task_name == 'hit':
            if self.index % (self.num_samples // 6) == 0:       # change object
                changeObj = True
            else:
                changeObj = False

            self.obj_type = self.env.reset_object(changeType=changeObj, sleep=REAL_TIME)

        elif self.task_name == 'stack' or self.task_name == 'collision':
            if self.index % (self.num_samples // 6) == 0:       # change target and moving object
                changeObj = True
                changeTarget = True

            elif self.index % (self.num_samples // 36) == 0:      # change moving object
                changeObj = True
                changeTarget = False

            else:
                changeObj = False
                changeTarget = False

            self.obj_type = self.env.reset_object(changeTargetType=changeTarget, changeType=changeObj, sleep=REAL_TIME)

    def collect(self):
        print("{} task data collection --------- START".format(self.task_name))
        for ep in range(self.num_samples):
            self.run_episode()

        self.env.close()
        # indices = np.random.permutation(len(self.action_data))
        # np.save(os.path.join(self.path, "{}-task-states-img.npy").format(self.task_name), self.state_img[indices])
        # np.save(os.path.join(self.path, "{}-task-effects-img.npy").format(self.task_name), self.effect_img[indices])
        # np.save(os.path.join(self.path, "{}-task-states-pose.npy").format(self.task_name), self.state_pose[indices])
        # np.save(os.path.join(self.path, "{}-task-effects-pose.npy").format(self.task_name), self.effect_pose[indices])
        # np.save(os.path.join(self.path, "{}-task-actions.npy").format(self.task_name), self.action_data[indices])
        # print("{} task data collection --------- END".format(self.task_name))
