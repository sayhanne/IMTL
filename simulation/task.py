import os

import numpy as np

from environment import PushEnv, StackEnv, HitEnv  # , CollisionEnv
# import math

REAL_TIME = False


class TableTopTask:
    def __init__(self, data_path, N, task_name, gui=0):
        """ total 6 objs => 6 dim for one-hot obj id """
        total_obj = 6
        in_dim = 9     # pos (x, y, z), euler ori (sinx, cosx, siny, cosy, sinz, cosz)
        action_dim = 2 + total_obj  # sin (x), cos (x), obj_id
        out_dim = 9    # pos (x, y, z), euler ori (sinx, cosx, siny, cosy, sinz, cosz) OR displacement of pos and ori
        if task_name == 'push':
            self.env = PushEnv(gui=gui)
            self.env.initialize()

        elif task_name == 'stack':
            self.env = StackEnv(gui=gui)
            self.env.initialize()
            action_dim = total_obj * 2
            in_dim *= 2
            out_dim *= 2

        elif task_name == 'hit':
            self.env = HitEnv(gui=gui)
            self.env.initialize()

        # elif task_name == 'collision':
        #     TODO: fix here
        #     self.env = CollisionEnv(gui=gui)
        #     self.env.initialize()

        self.cursor = 0
        self.task_name = task_name
        self.angle = np.arange(180.)

        # start collecting data set
        self.path = data_path
        self.state_img = np.zeros((int(N), 256, 256), dtype=np.uint8)  # state (256 x 256 segmentation image)
        self.effect_img = np.zeros((int(N), 256, 256), dtype=np.uint8)  # effect (256 x 256 segmentation image)

        self.state_pose = np.zeros((int(N), in_dim))
        self.effect_pose = np.zeros((int(N), out_dim))
        self.effect_delta = np.zeros((int(N), out_dim))  # displacement of the object

        self.action_data = np.zeros((int(N), action_dim))

        self.num_samples = int(N)
        self.index = 0
        self.collect()

    def run_episode(self):
        chosen_angle = self.angle[self.cursor]
        # print(chosen_angle)
        action, (img_pre, state_pre),  (img_post, state_post), pose_delta = self.env.step(angle=chosen_angle,
                                                                                          sleep=REAL_TIME)
        self.state_img[self.index] = img_pre
        self.state_pose[self.index] = state_pre
        self.effect_img[self.index] = img_post
        self.effect_pose[self.index] = state_post
        self.effect_delta[self.index] = pose_delta
        self.action_data[self.index] = action
        self.index += 1
        if self.index == self.num_samples:
            return

        self.changeTypes()

    def changeTypes(self):
        if self.task_name == 'push' or self.task_name == 'hit':
            if self.index % (self.num_samples // 6) == 0:  # change object
                changeObj = True
                self.cursor = 0
            else:
                self.cursor += 1
                changeObj = False

            self.env.reset_object(changeObj=changeObj, sleep=REAL_TIME)

        elif self.task_name == 'stack':  # or self.task_name == 'collision':
            if self.index % (self.num_samples // 6) == 0:  # change target and moving object
                changeObj = True
                changeTarget = True

            elif self.index % (self.num_samples // 36) == 0:  # change moving object
                changeObj = True
                changeTarget = False

            else:
                changeObj = False
                changeTarget = False

            self.env.reset_object(changeTargetObj=changeTarget, changeObj=changeObj, sleep=REAL_TIME)

    def collect(self):
        print("{} task data collection --------- START".format(self.task_name))
        for ep in range(self.num_samples):
            self.run_episode()

        self.env.close()
        indices = np.random.permutation(len(self.action_data))  # Shuffle data
        np.save(os.path.join(self.path, "{}-task-states-img.npy").format(self.task_name), self.state_img[indices])
        np.save(os.path.join(self.path, "{}-task-effects-img.npy").format(self.task_name), self.effect_img[indices])
        np.save(os.path.join(self.path, "{}-task-states-pose.npy").format(self.task_name), self.state_pose[indices])
        np.save(os.path.join(self.path, "{}-task-effects-pose.npy").format(self.task_name), self.effect_pose[indices])
        np.save(os.path.join(self.path, "{}-task-effects-delta.npy").format(self.task_name), self.effect_delta[indices])
        np.save(os.path.join(self.path, "{}-task-actions.npy").format(self.task_name), self.action_data[indices])
        print("{} task data collection --------- END".format(self.task_name))
