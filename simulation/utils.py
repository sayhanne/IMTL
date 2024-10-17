import pkgutil

from pybullet_utils import bullet_client
from scipy.spatial.transform import Rotation
# import matplotlib.pyplot as plt
import numpy as np


def connect(gui=1):
    if gui:
        p = bullet_client.BulletClient(connection_mode=bullet_client.pybullet.GUI)
    else:
        p = bullet_client.BulletClient(connection_mode=bullet_client.pybullet.DIRECT)
        egl = pkgutil.get_loader("eglRenderer")
        if egl:
            p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
        else:
            p.loadPlugin("eglRendererPlugin")
    return p


def create_object(p, obj_type, size, position, rotation=[0, 0, 0], mass=1, color=None, with_link=False,
                  restitution=False, friction=False):
    collisionId = -1
    visualId = -1

    if obj_type == p.GEOM_SPHERE:
        collisionId = p.createCollisionShape(shapeType=obj_type, radius=size[0])

        if color == "random":
            color = np.random.rand(3).tolist() + [1]
            visualId = p.createVisualShape(shapeType=obj_type, radius=size[0], rgbaColor=color)
        elif color is not None:
            visualId = p.createVisualShape(shapeType=obj_type, radius=size[0], rgbaColor=color)

    elif obj_type in [p.GEOM_CAPSULE, p.GEOM_CYLINDER]:
        collisionId = p.createCollisionShape(shapeType=obj_type, radius=size[0], height=size[1])

        if color == "random":
            color = np.random.rand(3).tolist() + [1]
            visualId = p.createVisualShape(shapeType=obj_type, radius=size[0], length=size[1], rgbaColor=color)
        elif color is not None:
            visualId = p.createVisualShape(shapeType=obj_type, radius=size[0], length=size[1], rgbaColor=color)

    elif obj_type == p.GEOM_BOX:
        collisionId = p.createCollisionShape(shapeType=obj_type, halfExtents=size)

        if color == "random":
            color = np.random.rand(3).tolist() + [1]
            visualId = p.createVisualShape(shapeType=obj_type, halfExtents=size, rgbaColor=color)
        elif color is not None:
            visualId = p.createVisualShape(shapeType=obj_type, halfExtents=size, rgbaColor=color)

    elif obj_type == "random":
        obj = "%03d" % np.random.randint(1000)
        obj_id = p.loadURDF(f"random_urdfs/{obj}/{obj}.urdf", basePosition=position,
                            baseOrientation=p.getQuaternionFromEuler(rotation))
        return obj_id

    if with_link:
        obj_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=-1,
                                   basePosition=position, baseOrientation=p.getQuaternionFromEuler(rotation),
                                   linkMasses=[mass], linkCollisionShapeIndices=[collisionId],
                                   linkVisualShapeIndices=[visualId], linkPositions=[[0, 0, 0]],
                                   linkOrientations=[[0, 0, 0, 1]], linkInertialFramePositions=[[0, 0, 0]],
                                   linkInertialFrameOrientations=[[0, 0, 0, 1]], linkParentIndices=[0],
                                   linkJointTypes=[p.JOINT_FIXED], linkJointAxis=[[0, 0, 0]])
    else:
        obj_id = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=collisionId, baseVisualShapeIndex=visualId,
                                   basePosition=position, baseOrientation=p.getQuaternionFromEuler(rotation),)

        if restitution:
            p.changeDynamics(obj_id, -1, linearDamping=0, angularDamping=0,
                             rollingFriction=0.0001, spinningFriction=0.0001, restitution=0.8,
                             contactProcessingThreshold=0)
        if friction:
            # TODO: change here!!!!
            p.changeDynamics(obj_id, -1, lateralFriction=1, restitution=0.)
    return obj_id


def create_tabletop(p):
    objects = {"base": create_object(p, p.GEOM_BOX, mass=0, size=[0.15, 0.15, 0.2],
                                     position=[0., 0., 0.2], color=[0.5, 0.5, 0.5, 1.0], with_link=True),
               "table": create_object(p, p.GEOM_BOX, mass=0, size=[0.4, 0.4, 0.2],
                                      position=[0.9, 0, 0.2], color=[0.9, 0.9, 0.9, 1.0], friction=True),
               "wall1": create_object(p, p.GEOM_BOX, mass=0, size=[0.4, 0.01, 0.05],
                                      position=[0.9, -0.39, 0.45], color=[1.0, 0.6, 0.6, 1.0], restitution=True),
               "wall2": create_object(p, p.GEOM_BOX, mass=0, size=[0.4, 0.01, 0.05],
                                      position=[0.9, 0.39, 0.45], color=[1.0, 0.6, 0.6, 1.0], restitution=True),
               "wall3": create_object(p, p.GEOM_BOX, mass=0, size=[0.01, 0.4, 0.05],
                                      position=[0.9 - 0.39, 0., 0.45], color=[1.0, 0.6, 0.6, 1.0], restitution=True),
               "wall4": create_object(p, p.GEOM_BOX, mass=0, size=[0.01, 0.4, 0.05],
                                      position=[0.9 + 0.39, 0., 0.45], color=[1.0, 0.6, 0.6, 1.0], restitution=True)}
    # walls
    return objects


def get_image(p, height, width):
    viewMatrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.9, 0., 0.45],
                                                     distance=0.87,
                                                     yaw=90,
                                                     pitch=-90,
                                                     roll=0,
                                                     upAxisIndex=2)
    far = 5.5
    near = 0.75
    projectionMatrix = p.computeProjectionMatrixFOV(fov=45, aspect=1.0, nearVal=near, farVal=far)
    _, _, rgb, depth, seg = p.getCameraImage(height=height, width=width, viewMatrix=viewMatrix,
                                             projectionMatrix=projectionMatrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
    depth_buffer_opengl = np.reshape(depth, [width, height])
    depth_opengl = far * near / (far - (far - near) * depth_buffer_opengl)
    return rgb[:, :, :3], depth_opengl, seg


def create_camera(p, position, rotation, static=True):
    baseCollision = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02])
    targetCollision = p.createCollisionShape(shapeType=p.GEOM_CYLINDER, radius=0.005, height=0.01)
    baseVisual = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02], rgbaColor=[0, 0, 0, 1])
    targetVisual = p.createVisualShape(shapeType=p.GEOM_CYLINDER, radius=0.005, length=0.01,
                                       rgbaColor=[0.8, 0.8, 0.8, 1.0])

    mass = 0 if static else 0.1
    obj_id = p.createMultiBody(baseMass=mass,
                               baseCollisionShapeIndex=-1,
                               baseVisualShapeIndex=-1,
                               basePosition=position,
                               baseOrientation=p.getQuaternionFromEuler(rotation),
                               linkMasses=[mass, mass],
                               linkCollisionShapeIndices=[baseCollision, targetCollision],
                               linkVisualShapeIndices=[baseVisual, targetVisual],
                               linkPositions=[[0, 0, 0], [0.02, 0, 0]],
                               linkOrientations=[[0, 0, 0, 1], p.getQuaternionFromEuler([0., np.pi / 2, 0])],
                               linkInertialFramePositions=[[0, 0, 0], [0, 0, 0]],
                               linkInertialFrameOrientations=[[0, 0, 0, 1], [0, 0, 0, 1]],
                               linkParentIndices=[0, 1],
                               linkJointTypes=[p.JOINT_FIXED, p.JOINT_FIXED],
                               linkJointAxis=[[0, 0, 0], [0, 0, 0]])

    return obj_id


def get_image_from_cam(p, camera_id, height, width):
    cam_state = p.getLinkStates(camera_id, [0, 1])
    base_pos = cam_state[0][0]
    up_vector = Rotation.from_quat(cam_state[0][1]).as_matrix()[:, -1]
    target_pos = cam_state[1][0]
    target_vec = np.array(target_pos) - np.array(base_pos)
    target_vec = (target_vec / np.linalg.norm(target_vec))
    return get_image(base_pos + target_vec * 0.04, base_pos + target_vec, up_vector, height, width)
