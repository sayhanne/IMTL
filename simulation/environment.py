import copy
import math
import pybullet_data
import numpy as np

import utils
import manipulators


class GenericEnv:
    def __init__(self, gui=0, seed=None):
        self._p = utils.connect(gui)
        self.gui = gui
        self.reset(seed=seed)
        """
        6 obj categories:
        sphere
        box
        cylinder (vertical and horizontal)
        prism   (vertical and horizontal)

        
        pybullet objects
        p.GEOM_BOX = 3
        p.GEOM_CAPSULE = 7
        p.GEOM_CYLINDER = 4 (vertical)
        p.GEOM_MESH = 5
        p.GEOM_PLANE = 6
        p.GEOM_SPHERE = 2
        
        self defined objects
        8 = cylinder (horizontal)
        9 = square prism (vertical)
        10 = square prism (horizontal)
        """

        self.encoded_ids = {2: [1, 0, 0, 0, 0, 0],
                            3: [0, 1, 0, 0, 0, 0],
                            4: [0, 0, 1, 0, 0, 0],
                            8: [0, 0, 0, 1, 0, 0],
                            9: [0, 0, 0, 0, 1, 0],
                            10: [0, 0, 0, 0, 0, 1],
                            }
        self.objects = [2, 3, 4, 8, 9, 10]
        self.x_range = [0.7, 1.1]
        self.y_range = [-0.2, 0.2]
        self.z = 0.45
        self.num_steps = 240

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        self._p.resetSimulation()
        self._p.configureDebugVisualizer(self._p.COV_ENABLE_GUI, 0)
        self._p.configureDebugVisualizer(self._p.COV_ENABLE_SHADOWS, 0)
        # self._p.setPhysicsEngineParameter(restitutionVelocityThreshold=0, warmStartingFactor=0, useSplitImpulse=True,
        #                                   splitImpulsePenetrationThreshold=0, contactSlop=0)
        # self._p.setPhysicsEngineParameter(enableConeFriction=False)
        self._p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._p.setGravity(0, 0, -9.807)
        self.plane_id = self._p.loadURDF("plane.urdf")

        self.env_dict = utils.create_tabletop(self._p)
        self.agent = manipulators.Manipulator(p=self._p, path="ur10e/ur10e.urdf", position=[0., 0., 0.4], ik_idx=30)
        base_constraint = self._p.createConstraint(parentBodyUniqueId=self.env_dict["base"], parentLinkIndex=0,
                                                   childBodyUniqueId=self.agent.id, childLinkIndex=-1,
                                                   jointType=self._p.JOINT_FIXED, jointAxis=(0, 0, 0),
                                                   parentFramePosition=(0, 0, 0),
                                                   childFramePosition=(0.0, 0.0, -0.2),
                                                   childFrameOrientation=(0, 0, 0, 1))
        self._p.changeConstraint(base_constraint, maxForce=10000)
        # force grippers to act in sync
        mimic_constraint = self._p.createConstraint(self.agent.id, 28, self.agent.id, 29,
                                                    jointType=self._p.JOINT_GEAR,
                                                    jointAxis=[1, 0, 0],
                                                    parentFramePosition=[0, 0, 0],
                                                    childFramePosition=[0, 0, 0])
        self._p.changeConstraint(mimic_constraint, gearRatio=-1, erp=0.1, maxForce=50)

    def init_agent_pose(self, t=None, sleep=False, traj=False):
        angles = [-0.294, -1.950, 2.141, -2.062, -1.572, 1.277]
        self.agent.set_joint_position(angles, t=t, sleep=sleep, traj=traj)

    def init_object(self, obj, position=None, orientation=None):
        if not position:
            position = [np.random.uniform(self.x_range[0], self.x_range[1], size=1),
                        np.random.uniform(self.y_range[0], self.y_range[1], size=1), self.z]
        color = None
        size = [0.03, 0.03, 0.03]
        mass = 1
        # box_dynamics = {"restitution": 0.,
        #                 "lateralFriction": 0.3,
        #                 "rollingFriction": 0.001,
        #                 "spinningFriction": 0.001,
        #                 "linearDamping": 0.01,
        #                 "angularDamping": 0.01,
        #                 "contactProcessingThreshold": 0}

        spherical_dynamics = {"restitution": 0.8,
                              "lateralFriction": 0.3,
                              "rollingFriction": 0.0001,
                              "spinningFriction": 0.0001,
                              "linearDamping": 0.01,
                              "angularDamping": 0.01,
                              "contactProcessingThreshold": 0}

        dynamics = None

        if obj == self._p.GEOM_SPHERE:  # 2
            color = [0.8, 0., 0.8, 1.]  # purple
            dynamics = spherical_dynamics

        elif obj == self._p.GEOM_CYLINDER:  # 4
            size = [0.03, 0.09]
            color = [0.8, 0., 0., 1.]  # red
            dynamics = spherical_dynamics

        elif obj == 8:  # horizontal cylinder (rollable)
            obj = self._p.GEOM_CYLINDER
            size = [0.03, 0.09]
            color = [0.8, 0.8, 0., 1.]  # yellow
            if not orientation:
                orientation = [np.pi / 2, 0, 0]
            # prevent rolling on its own
            spherical_dynamics["rollingFriction"] = 0.0002
            spherical_dynamics["spinningFriction"] = 0.0002
            dynamics = spherical_dynamics

        elif obj == self._p.GEOM_BOX:  # 3
            color = [0., 0.8, 0., 1.]  # green
            # dynamics = box_dynamics

        elif obj == 9:  # vertical square prism
            obj = self._p.GEOM_BOX
            size = [0.03, 0.03, 0.06]
            color = [0., 0.8, 0.8, 1.]  # cyan
            # dynamics = box_dynamics

        elif obj == 10:  # horizontal square prism
            obj = self._p.GEOM_BOX
            size = [0.03, 0.03, 0.06]
            color = [0., 0., 0.8, 1.]  # blue
            if not orientation:
                orientation = [np.pi / 2, 0, 0]
            # dynamics = box_dynamics

        if not orientation:
            orientation = [0., 0., 0.]

        obj_id = utils.create_object(p=self._p, obj_type=obj, size=size,
                                     position=position,
                                     rotation=orientation, color=color, mass=mass,
                                     dynamics=dynamics)
        return obj_id, orientation

    def get_obj_info(self, obj=False):
        raise NotImplementedError

    def state(self, obj=False):
        rgb, depth, seg = utils.get_image(p=self._p, height=256, width=256)
        obj_info = self.get_obj_info(obj=obj)
        return seg, obj_info

    """
    convert segmentation class from pybullet id to one-hot encoded obj id
    """

    def segment_obj_(self, seg):
        raise NotImplementedError

    def _step(self, count=1):
        for _ in range(count):
            self._p.stepSimulation()

    def __del__(self):
        self._p.disconnect()


class PushEnv(GenericEnv):
    def __init__(self, gui=0, seed=None):
        super(PushEnv, self).__init__(gui=gui, seed=seed)  # Reset Generic env

        self.obj_id = -1  # given pybullet id
        self.obj = -1  # will be updated in each iteration
        self.obj_index = 0  # index
        self.obj_ori_df = None

    def initialize(self):
        self.init_agent_pose(t=1)
        obj = self.objects[self.obj_index]
        self.obj = obj
        self.obj_id, self.obj_ori_df = self.init_object(obj=obj)
        self._step(self.num_steps)
        self.agent.close_gripper(1, sleep=True)

    def reset_object(self, changeObj=False, sleep=False):
        if changeObj:  # Switch objects sequentially
            self._p.removeBody(self.obj_id)
            self.obj_index += 1
            self.obj = self.objects[self.obj_index]
            self.obj_id, self.obj_ori_df = self.init_object(obj=self.obj)
        else:
            self._p.resetBasePositionAndOrientation(self.obj_id,
                                                    [np.random.uniform(self.x_range[0], self.x_range[1], size=1),
                                                     np.random.uniform(self.y_range[0], self.y_range[1], size=1),
                                                     self.z],
                                                    self._p.getQuaternionFromEuler(self.obj_ori_df))
        self.agent._waitsleep(0.5, sleep=sleep)

    def get_obj_info(self, obj=False):
        pose = np.zeros(9, dtype=np.float32)
        pos, quat = self._p.getBasePositionAndOrientation(self.obj_id)
        euler_angles = self._p.getEulerFromQuaternion(quat)
        pose[:3] = pos
        pose[3:] = [np.cos(euler_angles[0]), np.sin(euler_angles[0]),
                    np.cos(euler_angles[1]), np.sin(euler_angles[1]),
                    np.cos(euler_angles[2]), np.sin(euler_angles[2])]

        if obj:
            return {'object': np.hstack((pose, self.encoded_ids[self.obj]))}
        else:
            return {'object': pose}

    def segment_obj_(self, seg):
        seg[seg == self.env_dict["table"]] = 0  # background
        seg[seg == self.obj_id] = np.argmax(self.encoded_ids[self.obj]) + 1  # object

    def change_ori_angle(self, pos, angle):
        orientation = [self.obj_ori_df[0], self.obj_ori_df[1], math.radians(angle)]
        self._p.resetBasePositionAndOrientation(self.obj_id,
                                                pos,
                                                self._p.getQuaternionFromEuler(orientation))

    def step(self, angle, sleep=False, margin=0., dist_before=10, distance_after=5, contact_time=0.2):
        obj_info = self.get_obj_info()['object']
        obj_loc = obj_info[:3]

        """
        change ori if the contact surface is not spherical
        """
        if self.obj != self._p.GEOM_SPHERE and self.obj != self._p.GEOM_CYLINDER:
            self.change_ori_angle(obj_loc, angle)

        if self.obj == 9:  # vertical prism object
            if type(self) is PushEnv:
                margin = 0.02
            elif type(self) is HitEnv:
                margin *= 2

        img_pre, state_pre = self.state()
        """ inplace operation below, be careful"""
        self.segment_obj_(img_pre)
        state_pre = state_pre['object']  # get only value

        pos = [0., 0., 0.]
        pos[0] = obj_loc[0] - dist_before * math.sin(math.radians(angle - 90)) * 0.01
        pos[1] = obj_loc[1] + dist_before * math.cos(math.radians(angle - 90)) * 0.01
        pos[2] = obj_loc[2] + 0.1  # offset to avoid collision before push action

        if angle > 150:
            angle_rotate = angle - 180
        else:
            angle_rotate = angle

        quat = self._p.getQuaternionFromEuler([np.pi, 0, math.radians(angle_rotate)])
        self.agent.move_in_cartesian(pos, quat, t=1, sleep=sleep)

        pos[2] = obj_loc[2] + margin  # final push position
        self.agent.move_in_cartesian(pos, quat, t=0.5, sleep=sleep)

        pos[0] = obj_loc[0] + distance_after * math.sin(math.radians(angle - 90)) * 0.01
        pos[1] = obj_loc[1] - distance_after * math.cos(math.radians(angle - 90)) * 0.01
        self.agent.move_in_cartesian(pos, quat, t=contact_time, sleep=sleep)

        pos[2] = obj_loc[2] + 0.2
        self.agent.move_in_cartesian(pos, quat, t=1, sleep=sleep)
        self.init_agent_pose(t=0.25, sleep=sleep)
        self.agent._waitsleep(1, sleep=sleep)
        img_post, state_post = self.state()
        """ inplace operation below, be careful"""
        self.segment_obj_(img_post)
        state_post = state_post['object']  # get only value
        pose_delta = state_post - state_pre  # pos and ori
        action = np.hstack((self.encoded_ids[self.obj], [math.sin(math.radians(angle)), math.cos(math.radians(angle))]))
        return action, (img_pre, state_pre), (img_post, state_post), pose_delta

    def close(self):
        self._p.removeBody(self.agent.id)
        self._p.removeBody(self.obj_id)
        for key in self.env_dict:
            obj_id = self.env_dict[key]
            self._p.removeBody(obj_id)
        self._p.removeBody(self.plane_id)


class HitEnv(PushEnv):
    def __init__(self, gui=0, seed=None):
        super(HitEnv, self).__init__(gui=gui, seed=seed)

    def step(self, angle, sleep=False, margin=0.02, dist_before=10, distance_after=5, contact_time=0.1):
        return super().step(angle, sleep=sleep, margin=margin, contact_time=contact_time)


class StackEnv(GenericEnv):
    def __init__(self, gui=0, seed=None):
        super(StackEnv, self).__init__(gui=gui, seed=seed)  # Reset Generic env

        # Two objects in the environment
        self.target_id = -1
        self.obj_id = -1

        self.target_obj = -1
        self.obj = -1

        self.target_index = 0
        self.obj_index = 0

        # self.ds = 0.075
        # self.debug_items = []
        self.traj_t = 1.5

    def random_ori(self, obj):
        x_ori = 0.
        y_ori = 0.
        z_ori = np.random.uniform(0., np.pi)
        if obj == 8 or obj == 10:  # horizontal objects
            x_ori = np.pi / 2

        orientation = [x_ori, y_ori, z_ori]
        return orientation  # euler

    def initialize(self):
        self.init_agent_pose(t=1)
        target_obj = self.objects[self.target_index]  # target object
        obj = self.objects[self.obj_index]  # moving object
        self.target_obj = target_obj
        self.obj = obj

        self.target_id, _ = self.init_object(obj=target_obj,
                                             orientation=self.random_ori(self.target_obj))
        self.obj_id, _ = self.init_object(obj=obj, orientation=self.random_ori(self.obj))

        while True:  # check collision
            contacts = self._p.getClosestPoints(self.target_id, self.obj_id, distance=0.05)
            # If there are no collisions, break the loop
            if len(contacts) == 0:
                break

            self._p.resetBasePositionAndOrientation(self.obj_id,
                                                    [np.random.uniform(self.x_range[0], self.x_range[1], size=1),
                                                     np.random.uniform(self.y_range[0], self.y_range[1], size=1),
                                                     self.z],
                                                    self._p.getQuaternionFromEuler(self.random_ori(self.obj)))

        self._step(self.num_steps)
        self.agent.open_gripper(1, sleep=True)

    def reset_object(self, changeTargetObj=False, changeObj=False, sleep=False):
        if changeTargetObj:
            self._p.removeBody(self.target_id)
            self.target_index += 1
            self.target_obj = self.objects[self.target_index]
            self.target_id, _ = self.init_object(obj=self.target_obj,
                                                 orientation=self.random_ori(self.target_obj))

            self._p.removeBody(self.obj_id)
            self.obj_index = 0
            self.obj = self.objects[self.obj_index]
            self.obj_id, _ = self.init_object(obj=self.obj,
                                              orientation=self.random_ori(self.obj))

        elif changeObj:
            self._p.removeBody(self.obj_id)
            self.obj_index += 1
            self.obj = self.objects[self.obj_index]
            self.obj_id, _ = self.init_object(obj=self.obj,
                                              orientation=self.random_ori(self.obj))

            # do not change target type but change position & orientation
            self._p.resetBasePositionAndOrientation(self.target_id,
                                                    [np.random.uniform(self.x_range[0], self.x_range[1], size=1),
                                                     np.random.uniform(self.y_range[0], self.y_range[1], size=1),
                                                     self.z],
                                                    self._p.getQuaternionFromEuler(self.random_ori(self.target_obj)))

        else:  # new random positions
            self._p.resetBasePositionAndOrientation(self.target_id,
                                                    [np.random.uniform(self.x_range[0], self.x_range[1], size=1),
                                                     np.random.uniform(self.y_range[0], self.y_range[1], size=1),
                                                     self.z],
                                                    self._p.getQuaternionFromEuler(self.random_ori(self.target_obj)))

            self._p.resetBasePositionAndOrientation(self.obj_id,
                                                    [np.random.uniform(self.x_range[0], self.x_range[1], size=1),
                                                     np.random.uniform(self.y_range[0], self.y_range[1], size=1),
                                                     self.z],
                                                    self._p.getQuaternionFromEuler(self.random_ori(self.obj)))

        while True:
            contacts = self._p.getClosestPoints(self.target_id, self.obj_id, distance=0.05)
            # If there are no collisions, break the loop
            if len(contacts) == 0:
                break

            self._p.resetBasePositionAndOrientation(self.obj_id,
                                                    [np.random.uniform(self.x_range[0], self.x_range[1], size=1),
                                                     np.random.uniform(self.y_range[0], self.y_range[1], size=1),
                                                     self.z],
                                                    self._p.getQuaternionFromEuler(self.random_ori(self.obj)))

        self.agent._waitsleep(0.5, sleep=sleep)

    def get_obj_info(self, obj=False):
        target_pose = np.zeros(9, dtype=np.float32)
        obj_pose = np.zeros(9, dtype=np.float32)

        target_loc, target_quat = self._p.getBasePositionAndOrientation(self.target_id)
        obj_loc, obj_quat = self._p.getBasePositionAndOrientation(self.obj_id)
        euler_angles_target = self._p.getEulerFromQuaternion(target_quat)
        euler_angles_obj = self._p.getEulerFromQuaternion(obj_quat)

        target_pose[:3] = target_loc
        target_pose[3:] = [np.cos(euler_angles_target[0]), np.sin(euler_angles_target[0]),
                           np.cos(euler_angles_target[1]), np.sin(euler_angles_target[1]),
                           np.cos(euler_angles_target[2]), np.sin(euler_angles_target[2])]
        obj_pose[:3] = obj_loc
        obj_pose[3:] = [np.cos(euler_angles_obj[0]), np.sin(euler_angles_obj[0]),
                        np.cos(euler_angles_obj[1]), np.sin(euler_angles_obj[1]),
                        np.cos(euler_angles_obj[2]), np.sin(euler_angles_obj[2])]

        if obj:
            return {'target': np.hstack((target_pose, self.encoded_ids[self.target_obj])),
                    'object': np.hstack((obj_pose, self.encoded_ids[self.obj]))}
        else:
            return {'target': target_pose,
                    'object': obj_pose}

    def segment_obj_(self, seg):  # inplace modification!!!
        seg[seg == self.env_dict["table"]] = 0  # background
        seg[seg == self.target_id] = np.argmax(self.encoded_ids[self.target_obj]) + 1  # target object
        seg[seg == self.obj_id] = np.argmax(self.encoded_ids[self.obj]) + 1  # object

    def step(self, angle, sleep=False):
        _, obj_quat = self._p.getBasePositionAndOrientation(self.obj_id)
        img_pre, state_pre = self.state()
        """ inplace operation below, be careful"""
        self.segment_obj_(img_pre)

        grap_obj_loc = state_pre['object'][:3]
        grap_ori_euler = self._p.getEulerFromQuaternion(obj_quat)
        target_obj_loc = state_pre['target'][:3]

        state_pre = np.hstack((state_pre['target'], state_pre['object']))  # get only value
        quat1 = self._p.getQuaternionFromEuler([np.pi, 0., grap_ori_euler[2] - np.pi / 2])
        grap_obj_loc[2] -= 0.01
        target_obj_loc[2] -= 0.01

        up_pos_1 = copy.deepcopy(grap_obj_loc)
        up_pos_1[2] = 0.9

        up_pos_2 = copy.deepcopy(target_obj_loc)
        up_pos_2[2] = 0.9

        self.agent.move_in_cartesian(up_pos_1, orientation=quat1, t=self.traj_t, sleep=sleep)
        self.agent.move_in_cartesian(grap_obj_loc, orientation=quat1, t=self.traj_t, sleep=sleep)

        self.agent.close_gripper(t=0.1, sleep=sleep)
        self.agent.move_in_cartesian(up_pos_1, orientation=quat1, t=self.traj_t, sleep=sleep)

        self.agent.move_in_cartesian(up_pos_1, orientation=quat1, t=self.traj_t, sleep=sleep)
        self.agent.move_in_cartesian(up_pos_2, orientation=quat1, t=self.traj_t, sleep=sleep)

        self.agent.move_in_cartesian(target_obj_loc, orientation=quat1, t=self.traj_t, sleep=sleep)
        self.agent.open_gripper(t=0.1, sleep=sleep)
        self.agent.move_in_cartesian(up_pos_2, orientation=quat1, t=self.traj_t, sleep=sleep)

        self.init_agent_pose(0.25, sleep=sleep)
        self.agent._waitsleep(1, sleep=sleep)
        img_post, state_post = self.state()
        """ inplace operation below, be careful"""
        self.segment_obj_(img_post)
        state_post = np.hstack((state_post['target'], state_post['object']))  # get only value

        pose_delta = np.hstack((state_post[:9] - state_pre[:9],  # target obj displacement (pos, ori)
                                state_post[9:] - state_pre[9:]))  # moving obj displacement (pos, ori)

        action = np.hstack((self.encoded_ids[self.target_obj], self.encoded_ids[self.obj]))

        return action, (img_pre, state_pre), (
            img_post, state_post), pose_delta

    def close(self):
        self._p.removeBody(self.agent.id)
        self._p.removeBody(self.target_id)
        self._p.removeBody(self.obj_id)
        for key in self.env_dict:
            obj_id = self.env_dict[key]
            self._p.removeBody(obj_id)
        self._p.removeBody(self.plane_id)


"""
An environment which includes two objects. One object will be pushed towards the other target object
and the collision effect will be observed.
Object that is going to be pushed will be created during the step function in order to calculate the
right position for it.
"""

# class CollisionEnv(GenericEnv):
#     # TODO: update it
#     def __init__(self, gui=0, seed=None):
#         super(CollisionEnv, self).__init__(gui=gui, seed=seed)
#         # Two objects in the environment
#         self.target_id = -1
#         self.obj_id = -1
#         self.obj_ori_df = [0., 0., 0.]
#         self.target_ori_df = [0., 0., 0.]
#
#         self.target_type = -1
#         self.obj_type = -1
#
#         self.target_index = 0
#         self.obj_index = 0
#         self.traj_t = 1.5
#
#     def initialize(self):
#         self.init_agent_pose(t=1)
#         target_type = self.obj_types[self.target_index]  # target object
#         obj_type = self.obj_types[self.obj_index]  # moving object
#         self.target_type = target_type
#         self.obj_type = obj_type
#         self.target_id, self.target_ori_df = self.init_object(obj_type=target_type)
#         self._step(self.num_steps)
#         self.agent.close_gripper(1, sleep=True)
#         return np.hstack((self.encoded_ids[self.target_type], self.encoded_ids[self.obj_type]))
#
#     def reset_object(self, changeTargetType=False, changeType=False, sleep=False):
#         if changeTargetType:
#             self._p.removeBody(self.target_id)
#             self.target_index += 1
#             self.target_type = self.obj_types[self.target_index]
#             self.target_id, self.target_ori_df = self.init_object(obj_type=self.target_type)
#
#             self._p.removeBody(self.obj_id)
#             self.obj_index = 0
#             self.obj_type = self.obj_types[self.obj_index]
#
#         else:
#             if changeType:
#                 self.obj_index += 1
#                 self.obj_type = self.obj_types[self.obj_index]
#
#             self._p.removeBody(self.obj_id)
#             self._p.resetBasePositionAndOrientation(self.target_id,
#                                                     [np.random.uniform(self.x_range[0], self.x_range[1], size=1),
#                                                      np.random.uniform(self.y_range[0], self.y_range[1], size=1),
#                                                      self.z],
#                                                     self._p.getQuaternionFromEuler(self.target_ori_df))
#
#         self.agent._waitsleep(0.5, sleep=sleep)
#         return np.hstack((self.encoded_ids[self.target_type], self.encoded_ids[self.obj_type]))
#
#     def get_obj_pos(self):
#         target_pos, target_ori = self._p.getBasePositionAndOrientation(self.target_id)
#         obj_pos, obj_ori = self._p.getBasePositionAndOrientation(self.obj_id)
#         target_pos, obj_pos = list(target_pos), list(obj_pos)
#         return np.hstack((np.asarray(target_pos), np.asarray(obj_pos)))
#
#     def change_ori_action(self, pos, action):
#         orientation = [self.obj_ori_df[0], self.obj_ori_df[1], math.radians(action)]
#         self._p.resetBasePositionAndOrientation(self.obj_id,
#                                                 pos,
#                                                 self._p.getQuaternionFromEuler(orientation))
#
#     def step(self, action, sleep=False, obj_dist=10, distance_before=5, distance_after=5):
#         ee_pos = [0., 0., 0.]
#         calc_pos = [0., 0., 0.]
#         target_pos, _ = self._p.getBasePositionAndOrientation(self.target_id)
#         target_pos = list(target_pos)
#
#         # calculate obj position first
#         calc_pos[0] = target_pos[0] - obj_dist * math.sin(math.radians(action - 90)) * 0.01
#         calc_pos[1] = target_pos[1] + obj_dist * math.cos(math.radians(action - 90)) * 0.01
#         calc_pos[2] = self.z
#         self.obj_id, self.obj_ori_df = self.init_object(obj_type=self.obj_type, position=calc_pos)
#         self._step(self.num_steps)
#         img_pre, pos_pre = self.state()
#
#         obj_pos, _ = self._p.getBasePositionAndOrientation(self.obj_id)
#         obj_pos = list(obj_pos)
#
#         """
#         change ori if the contact surface is not spherical
#         """
#         if self.obj_type != self._p.GEOM_SPHERE and self.obj_type != self._p.GEOM_CYLINDER:
#             self.change_ori_action(obj_pos, action)
#
#         if self.obj_type == 9:  # vertical prism object
#             margin = 0.02
#         else:
#             margin = 0.
#
#         ee_pos[0] = obj_pos[0] - distance_before * math.sin(math.radians(action - 90)) * 0.01
#         ee_pos[1] = obj_pos[1] + distance_before * math.cos(math.radians(action - 90)) * 0.01
#         ee_pos[2] = obj_pos[2] + 0.1  # offset to avoid collision before push action
#
#         if action > 150:
#             action_rotate = action - 180
#         else:
#             action_rotate = action
#
#         self.agent.move_in_cartesian(ee_pos, self._p.getQuaternionFromEuler([np.pi, 0, math.radians(action_rotate)]),
#                                      t=self.traj_t, sleep=sleep)
#
#         ee_pos[2] = obj_pos[2] + margin
#         self.agent.move_in_cartesian(ee_pos, self._p.getQuaternionFromEuler([np.pi, 0, math.radians(action_rotate)]),
#                                      t=0.5, sleep=sleep)
#
#         ee_pos[0] = target_pos[0] + distance_after * math.sin(math.radians(action - 90)) * 0.01
#         ee_pos[1] = target_pos[1] - distance_after * math.cos(math.radians(action - 90)) * 0.01
#         self.agent.move_in_cartesian(ee_pos, self._p.getQuaternionFromEuler([np.pi, 0, math.radians(action_rotate)]),
#                                      t=0.5, sleep=sleep)
#
#         ee_pos[2] = target_pos[2] + 0.2
#         self.agent.move_in_cartesian(ee_pos, self._p.getQuaternionFromEuler([np.pi, 0, math.radians(action_rotate)]),
#                                      t=1, sleep=sleep)
#         self.init_agent_pose(t=0.25, sleep=sleep)
#         self.agent._waitsleep(1, sleep=sleep)
#         img_post, pos_post = self.state()
#         return (img_pre, pos_pre), (img_post, pos_post)
#
#     def close(self):
#         self._p.removeBody(self.agent.id)
#         self._p.removeBody(self.target_id)
#         self._p.removeBody(self.obj_id)
#         for key in self.env_dict:
#             obj_id = self.env_dict[key]
#             self._p.removeBody(obj_id)
#         self._p.removeBody(self.plane_id)
