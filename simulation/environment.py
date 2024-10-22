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
        4 obj categories:
        sphere
        box
        cylinder
        prism

        
        pybullet ids
        GEOM_BOX = 3
        GEOM_CAPSULE = 7
        GEOM_CYLINDER = 4 (vertical)
        GEOM_MESH = 5
        GEOM_PLANE = 6
        GEOM_SPHERE = 2
        
        self defined ids
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
        self.obj_types = [8, 9, 10, 2, 3, 4]
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

    def init_object(self, obj_type, position=None, orientation=None):
        if not position:
            position = [np.random.uniform(self.x_range[0], self.x_range[1], size=1),
                        np.random.uniform(self.y_range[0], self.y_range[1], size=1), self.z]

        if not orientation:
            orientation = [0., 0., 0.]
        color = None
        size = [0.03, 0.03, 0.03]
        mass = 1
        box_dynamics = {"restitution": 0.,
                        "lateralFriction": 0.3,
                        "rollingFriction": 0.001,
                        "spinningFriction": 0.001,
                        "linearDamping": 0.01,
                        "angularDamping": 0.01,
                        "contactProcessingThreshold": 0}

        spherical_dynamics = {"restitution": 0.8,
                              "lateralFriction": 0.3,
                              "rollingFriction": 0.0001,
                              "spinningFriction": 0.0001,
                              "linearDamping": 0.01,
                              "angularDamping": 0.01,
                              "contactProcessingThreshold": 0}

        dynamics = None

        if obj_type == self._p.GEOM_SPHERE:  # 2
            color = [0.8, 0., 0.8, 1.]  # purple
            dynamics = spherical_dynamics

        elif obj_type == self._p.GEOM_CYLINDER:  # 4
            size = [0.03, 0.09]
            color = [0.8, 0., 0., 1.]  # red
            dynamics = spherical_dynamics

        elif obj_type == 8:  # horizontal cylinder (rollable)
            obj_type = self._p.GEOM_CYLINDER
            size = [0.03, 0.09]
            color = [0.8, 0.8, 0., 1.]  # yellow
            orientation = [np.pi / 2, 0, 0]
            dynamics = spherical_dynamics

        elif obj_type == self._p.GEOM_BOX:  # 3
            color = [0., 0.8, 0., 1.]  # green
            # dynamics = box_dynamics

        elif obj_type == 9:  # vertical square prism
            obj_type = self._p.GEOM_BOX
            size = [0.03, 0.03, 0.06]
            color = [0., 0.8, 0.8, 1.]  # cyan
            # dynamics = box_dynamics

        elif obj_type == 10:  # horizontal square prism
            obj_type = self._p.GEOM_BOX
            size = [0.03, 0.03, 0.06]
            color = [0., 0., 0.8, 1.]  # blue
            orientation = [0, np.pi / 2, 0]
            # dynamics = box_dynamics

        obj_id = utils.create_object(p=self._p, obj_type=obj_type, size=size,
                                     position=position,
                                     rotation=orientation, color=color, mass=mass,
                                     dynamics=dynamics)
        return obj_id

    def get_obj_pos(self):
        pass

    def state(self):
        rgb, depth, seg = utils.get_image(p=self._p, height=256, width=256)
        poses = self.get_obj_pos()
        return depth, poses

    def _step(self, count=1):
        for _ in range(count):
            self._p.stepSimulation()

    def __del__(self):
        self._p.disconnect()


class PushEnv(GenericEnv):
    def __init__(self, gui=0, seed=None):
        super(PushEnv, self).__init__(gui=gui, seed=seed)  # Reset Generic env

        self.obj_id = -1  # will be updated in each iteration
        self.obj_type = -1  # will be updated in each iteration
        self.obj_index = 0  # index

    def initialize(self):
        self.init_agent_pose(t=1)
        obj_type = self.obj_types[self.obj_index]
        self.obj_type = obj_type
        self.obj_id = self.init_object(obj_type=obj_type)
        self._step(self.num_steps)
        self.agent.close_gripper(1, sleep=True)
        return self.encoded_ids[self.obj_type]

    def reset_object(self, changeType=False, sleep=False):
        if changeType:  # Switch objects sequentially
            self._p.removeBody(self.obj_id)
            self.obj_index += 1
            self.obj_type = self.obj_types[self.obj_index]
            self.obj_id = self.init_object(obj_type=self.obj_type)
        else:
            if self.obj_type == 8 or self.obj_type == 10:
                orientation = [np.pi / 2, 0, 0]  # horizontal placement
            else:
                orientation = [0., 0., 0.]
            self._p.resetBasePositionAndOrientation(self.obj_id,
                                                    [np.random.uniform(self.x_range[0], self.x_range[1], size=1),
                                                     np.random.uniform(self.y_range[0], self.y_range[1], size=1),
                                                     self.z],
                                                    self._p.getQuaternionFromEuler(orientation))
        self.agent._waitsleep(0.5, sleep=sleep)
        return self.encoded_ids[self.obj_type]

    def get_obj_pos(self):
        pos, ori = self._p.getBasePositionAndOrientation(self.obj_id)
        return list(pos)

    def get_obj_pos_ori(self):
        pos, ori = self._p.getBasePositionAndOrientation(self.obj_id)
        return list(pos), list(ori)

    def change_ori_action(self, obj_type, pos, action):
        if obj_type == 8 or self.obj_type == 10:
            orientation = [np.pi / 2, 0, math.radians(action)]
        else:
            orientation = [np.pi, 0., math.radians(action)]
        self._p.resetBasePositionAndOrientation(self.obj_id,
                                                pos,
                                                self._p.getQuaternionFromEuler(orientation))

    def step(self, action, sleep=False, margin=0., dist_before=10, distance_after=5, contact_time=0.2):
        obj_pos = self.get_obj_pos()
        """
        change ori if the contact surface is not spherical
        """
        if self.obj_type != self._p.GEOM_SPHERE and self.obj_type != self._p.GEOM_CYLINDER:
            self.change_ori_action(self.obj_type, obj_pos, action)

        if self.obj_type == 9:  # vertical prism object
            if type(self) is PushEnv:
                margin = 0.02
            elif type(self) is HitEnv:
                margin *= 2

        img_pre, pos_pre = self.state()

        pos = [0., 0., 0.]
        pos[0] = obj_pos[0] - dist_before * math.sin(math.radians(action - 90)) * 0.01
        pos[1] = obj_pos[1] + dist_before * math.cos(math.radians(action - 90)) * 0.01
        pos[2] = obj_pos[2] + 0.1  # offset to avoid collision before push action

        if action > 150:
            action_rotate = action - 180
        else:
            action_rotate = action

        self.agent.move_in_cartesian(pos, self._p.getQuaternionFromEuler([np.pi, 0, math.radians(action_rotate)]),
                                     t=1, sleep=sleep)

        pos[2] = obj_pos[2] + margin  # final push position
        self.agent.move_in_cartesian(pos, self._p.getQuaternionFromEuler([np.pi, 0, math.radians(action_rotate)]),
                                     t=0.5, sleep=sleep)

        pos[0] = obj_pos[0] + distance_after * math.sin(math.radians(action - 90)) * 0.01
        pos[1] = obj_pos[1] - distance_after * math.cos(math.radians(action - 90)) * 0.01
        self.agent.move_in_cartesian(pos, self._p.getQuaternionFromEuler([np.pi, 0, math.radians(action_rotate)]),
                                     t=contact_time, sleep=sleep)

        pos[2] = obj_pos[2] + 0.2
        self.agent.move_in_cartesian(pos, self._p.getQuaternionFromEuler([np.pi, 0, math.radians(action_rotate)]),
                                     t=1, sleep=sleep)
        self.init_agent_pose(t=0.25, sleep=sleep)
        self.agent._waitsleep(1, sleep=sleep)
        img_post, pos_post = self.state()
        return (img_pre, pos_pre), (img_post, pos_post)

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

    def step(self, action, sleep=False, margin=0.02, dist_before=10, distance_after=5, contact_time=0.1):
        return super().step(action, sleep=sleep, margin=margin, contact_time=contact_time)


"""
An environment which includes two objects. One object will be pushed towards the other target object
and the collision effect will be observed.
Object that is going to be pushed will be created during the step function in order to calculate the
right position for it.
"""


class CollisionEnv(GenericEnv):
    # TODO: fix it
    def __init__(self, gui=0, seed=None):
        super(CollisionEnv, self).__init__(gui=gui, seed=seed)
        # Two objects in the environment
        self.target_id = -1
        self.obj_id = -1

        self.target_type = -1
        self.obj_type = -1

        self.target_index = 0
        self.obj_index = 0
        self.traj_t = 1.5

    def initialize(self):
        self.init_agent_pose(t=1)
        target_type = self.obj_types[self.target_index]  # target object
        obj_type = self.obj_types[self.obj_index]  # moving object
        self.target_type = target_type
        self.obj_type = obj_type
        self.target_id = self.init_object(obj_type=target_type)
        self._step(self.num_steps)
        self.agent.close_gripper(1, sleep=True)
        return np.hstack((self.encoded_ids[self.target_type], self.encoded_ids[self.obj_type]))

    def reset_object(self, changeTargetType=False, changeType=False, sleep=False):
        if changeTargetType:
            self._p.removeBody(self.target_id)
            self.target_index += 1
            self.target_type = self.obj_types[self.target_index]
            self.target_id = self.init_object(obj_type=self.target_type)

            self._p.removeBody(self.obj_id)
            self.obj_index = 0
            self.obj_type = self.obj_types[self.obj_index]

        else:
            if changeType:
                self.obj_index += 1
                self.obj_type = self.obj_types[self.obj_index]

            self._p.removeBody(self.obj_id)
            if self.target_type == 8 or self.target_type == 10:
                orientation = [np.pi / 2, 0, 0]  # horizontal placement
            else:
                orientation = [0., 0., 0.]
            self._p.resetBasePositionAndOrientation(self.target_id,
                                                    [np.random.uniform(self.x_range[0], self.x_range[1], size=1),
                                                     np.random.uniform(self.y_range[0], self.y_range[1], size=1),
                                                     self.z],
                                                    self._p.getQuaternionFromEuler(orientation))

        self.agent._waitsleep(0.5, sleep=sleep)
        return np.hstack((self.encoded_ids[self.target_type], self.encoded_ids[self.obj_type]))

    def get_obj_pos(self):
        target_pos, target_ori = self._p.getBasePositionAndOrientation(self.target_id)
        obj_pos, obj_ori = self._p.getBasePositionAndOrientation(self.obj_id)
        target_pos, obj_pos = list(target_pos), list(obj_pos)
        return np.hstack((np.asarray(target_pos), np.asarray(obj_pos)))

    def change_ori_action(self, obj_type, pos, action):
        if obj_type == 8 or self.obj_type == 10:
            orientation = [np.pi / 2, 0, math.radians(action)]
        else:
            orientation = [np.pi, 0., math.radians(action)]
        self._p.resetBasePositionAndOrientation(self.obj_id,
                                                pos,
                                                self._p.getQuaternionFromEuler(orientation))

    def step(self, action, sleep=False, obj_dist=10, distance_before=5, distance_after=5):
        ee_pos = [0., 0., 0.]
        calc_pos = [0., 0., 0.]
        target_pos, _ = self._p.getBasePositionAndOrientation(self.target_id)
        target_pos = list(target_pos)

        # calculate obj position first
        calc_pos[0] = target_pos[0] - obj_dist * math.sin(math.radians(action - 90)) * 0.01
        calc_pos[1] = target_pos[1] + obj_dist * math.cos(math.radians(action - 90)) * 0.01
        calc_pos[2] = self.z
        self.obj_id = self.init_object(obj_type=self.obj_type, position=calc_pos)
        self._step(self.num_steps)
        img_pre, pos_pre = self.state()

        obj_pos, _ = self._p.getBasePositionAndOrientation(self.obj_id)
        obj_pos = list(obj_pos)

        """
        change ori if the contact surface is not spherical
        """
        if self.obj_type != self._p.GEOM_SPHERE and self.obj_type != self._p.GEOM_CYLINDER:
            self.change_ori_action(self.obj_type, obj_pos, action)

        if self.obj_type == 9:  # vertical prism object
            margin = 0.02
        else:
            margin = 0.

        ee_pos[0] = obj_pos[0] - distance_before * math.sin(math.radians(action - 90)) * 0.01
        ee_pos[1] = obj_pos[1] + distance_before * math.cos(math.radians(action - 90)) * 0.01
        ee_pos[2] = obj_pos[2] + 0.1  # offset to avoid collision before push action

        if action > 150:
            action_rotate = action - 180
        else:
            action_rotate = action

        self.agent.move_in_cartesian(ee_pos, self._p.getQuaternionFromEuler([np.pi, 0, math.radians(action_rotate)]),
                                     t=self.traj_t, sleep=sleep)

        ee_pos[2] = obj_pos[2] + margin
        self.agent.move_in_cartesian(ee_pos, self._p.getQuaternionFromEuler([np.pi, 0, math.radians(action_rotate)]),
                                     t=0.5, sleep=sleep)

        ee_pos[0] = target_pos[0] + distance_after * math.sin(math.radians(action - 90)) * 0.01
        ee_pos[1] = target_pos[1] - distance_after * math.cos(math.radians(action - 90)) * 0.01
        self.agent.move_in_cartesian(ee_pos, self._p.getQuaternionFromEuler([np.pi, 0, math.radians(action_rotate)]),
                                     t=0.5, sleep=sleep)

        ee_pos[2] = target_pos[2] + 0.2
        self.agent.move_in_cartesian(ee_pos, self._p.getQuaternionFromEuler([np.pi, 0, math.radians(action_rotate)]),
                                     t=1, sleep=sleep)
        self.init_agent_pose(t=0.25, sleep=sleep)
        self.agent._waitsleep(1, sleep=sleep)
        img_post, pos_post = self.state()
        return (img_pre, pos_pre), (img_post, pos_post)

    def close(self):
        self._p.removeBody(self.agent.id)
        self._p.removeBody(self.target_id)
        self._p.removeBody(self.obj_id)
        for key in self.env_dict:
            obj_id = self.env_dict[key]
            self._p.removeBody(obj_id)
        self._p.removeBody(self.plane_id)


class StackEnv(GenericEnv):
    def __init__(self, gui=0, seed=None):
        super(StackEnv, self).__init__(gui=gui, seed=seed)  # Reset Generic env

        # Two objects in the environment
        self.target_id = -1
        self.obj_id = -1

        self.target_type = -1
        self.obj_type = -1

        self.target_index = 0
        self.obj_index = 0

        # self.ds = 0.075
        # self.debug_items = []
        self.traj_t = 1.5

    def initialize(self):
        self.init_agent_pose(t=1)
        target_type = self.obj_types[self.target_index]  # target object
        obj_type = self.obj_types[self.obj_index]  # moving object
        self.target_type = target_type
        self.obj_type = obj_type
        self.target_id = self.init_object(obj_type=target_type)
        self.obj_id = self.init_object(obj_type=obj_type)
        target_orientation = [0., 0., 0.]
        obj_orientation = [0., 0., 0.]

        while True:  # check collision
            contacts = self._p.getClosestPoints(self.target_id, self.obj_id, distance=0.05)
            # If there are no collisions, break the loop
            if len(contacts) == 0:
                break
            self._p.resetBasePositionAndOrientation(self.target_id,
                                                    [np.random.uniform(self.x_range[0], self.x_range[1], size=1),
                                                     np.random.uniform(self.y_range[0], self.y_range[1], size=1),
                                                     self.z],
                                                    self._p.getQuaternionFromEuler(target_orientation))

            self._p.resetBasePositionAndOrientation(self.obj_id,
                                                    [np.random.uniform(self.x_range[0], self.x_range[1], size=1),
                                                     np.random.uniform(self.y_range[0], self.y_range[1], size=1),
                                                     self.z],
                                                    self._p.getQuaternionFromEuler(obj_orientation))

        self._step(self.num_steps)
        self.agent.open_gripper(1, sleep=True)
        return np.hstack((self.encoded_ids[self.target_type], self.encoded_ids[self.obj_type]))

    def reset_object(self, changeTargetType=False, changeType=False, sleep=False):
        if changeTargetType:
            self._p.removeBody(self.target_id)
            self.target_index += 1
            self.target_type = self.obj_types[self.target_index]
            self.target_id = self.init_object(obj_type=self.target_type)

            self._p.removeBody(self.obj_id)
            self.obj_index = 0
            self.obj_type = self.obj_types[self.obj_index]
            self.obj_id = self.init_object(obj_type=self.obj_type)

            target_orientation = [0., 0., 0.]
            obj_orientation = [0., 0., 0.]

            while True:  # check both of them
                self._p.resetBasePositionAndOrientation(self.target_id,
                                                        [np.random.uniform(self.x_range[0], self.x_range[1], size=1),
                                                         np.random.uniform(self.y_range[0], self.y_range[1], size=1),
                                                         self.z],
                                                        self._p.getQuaternionFromEuler(target_orientation))

                self._p.resetBasePositionAndOrientation(self.obj_id,
                                                        [np.random.uniform(self.x_range[0], self.x_range[1], size=1),
                                                         np.random.uniform(self.y_range[0], self.y_range[1], size=1),
                                                         self.z],
                                                        self._p.getQuaternionFromEuler(obj_orientation))

                contacts = self._p.getClosestPoints(self.target_id, self.obj_id, distance=0.05)
                # If there are no collisions, break the loop
                if len(contacts) == 0:
                    break

        elif changeType:
            self._p.removeBody(self.obj_id)
            self.obj_index += 1
            self.obj_type = self.obj_types[self.obj_index]
            self.obj_id = self.init_object(obj_type=self.obj_type)
            target_orientation = [0., 0., 0.]

            while True:  # check only target
                self._p.resetBasePositionAndOrientation(self.target_id,
                                                        [np.random.uniform(self.x_range[0], self.x_range[1], size=1),
                                                         np.random.uniform(self.y_range[0], self.y_range[1], size=1),
                                                         self.z],
                                                        self._p.getQuaternionFromEuler(target_orientation))

                contacts = self._p.getClosestPoints(self.target_id, self.obj_id, distance=0.05)
                # If there are no collisions, break the loop
                if len(contacts) == 0:
                    break

        else:
            target_orientation = [0., 0., 0.]
            obj_orientation = [0., 0., 0.]

            while True:  # check both of them
                self._p.resetBasePositionAndOrientation(self.target_id,
                                                        [np.random.uniform(self.x_range[0], self.x_range[1], size=1),
                                                         np.random.uniform(self.y_range[0], self.y_range[1], size=1),
                                                         self.z],
                                                        self._p.getQuaternionFromEuler(target_orientation))

                self._p.resetBasePositionAndOrientation(self.obj_id,
                                                        [np.random.uniform(self.x_range[0], self.x_range[1], size=1),
                                                         np.random.uniform(self.y_range[0], self.y_range[1], size=1),
                                                         self.z],
                                                        self._p.getQuaternionFromEuler(obj_orientation))

                contacts = self._p.getClosestPoints(self.target_id, self.obj_id, distance=0.05)
                # If there are no collisions, break the loop
                if len(contacts) == 0:
                    break

        self.agent._waitsleep(0.5, sleep=sleep)
        return np.hstack((self.encoded_ids[self.target_type], self.encoded_ids[self.obj_type]))

    def get_obj_pos(self):
        target_pos, target_ori = self._p.getBasePositionAndOrientation(self.target_id)
        obj_pos, obj_ori = self._p.getBasePositionAndOrientation(self.obj_id)
        target_pos, obj_pos = list(target_pos), list(obj_pos)
        return np.hstack((np.asarray(target_pos), np.asarray(obj_pos)))

    def step(self, action, sleep=False):
        img_pre, pos_pre = self.state()
        obj_locs = self.get_obj_pos()
        grap_obj_loc = obj_locs[3:]
        target_obj_loc = obj_locs[:3]

        _, obj_quat = self._p.getBasePositionAndOrientation(self.obj_id)
        grap_euler1 = self._p.getEulerFromQuaternion(obj_quat)
        # release_euler2 = [np.pi, 0, 0]

        quat1 = self._p.quat = self._p.getQuaternionFromEuler([np.pi, 0., grap_euler1[0]])
        # quat2 = self._p.getQuaternionFromEuler(release_euler2)
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
        img_post, pos_post = self.state()
        return (img_pre, pos_pre), (img_post, pos_post)

    def close(self):
        self._p.removeBody(self.agent.id)
        self._p.removeBody(self.target_id)
        self._p.removeBody(self.obj_id)
        for key in self.env_dict:
            obj_id = self.env_dict[key]
            self._p.removeBody(obj_id)
        self._p.removeBody(self.plane_id)
