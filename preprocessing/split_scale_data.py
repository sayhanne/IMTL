import pickle

import numpy as np
from sklearn.preprocessing import MinMaxScaler


def train_test_split(task, samples=1000):
    pre_pose = np.load('../data/{}-task-states-pose.npy'.format(task), allow_pickle=True)
    pre_img = np.load('../data/{}-task-states-img.npy'.format(task), allow_pickle=True)
    actions = np.load('../data/{}-task-actions.npy'.format(task), allow_pickle=True)
    post_pose = np.load('../data/{}-task-effects-pose.npy'.format(task), allow_pickle=True)
    post_img = np.load('../data/{}-task-effects-img.npy'.format(task), allow_pickle=True)
    delta_pose = np.load('../data/{}-task-effects-delta.npy'.format(task), allow_pickle=True)

    train_indices = np.random.choice(a=range(samples), size=samples*4//5, replace=False)
    test_indices = np.setdiff1d(range(samples), train_indices, assume_unique=True)

    train_pose_pre = np.take(pre_pose, train_indices, axis=0)
    train_img_pre = np.take(pre_img, train_indices, axis=0)
    train_actions = np.take(actions, train_indices, axis=0)
    train_pose_post = np.take(post_pose, train_indices, axis=0)
    train_img_post = np.take(post_img, train_indices, axis=0)
    train_pose_delta = np.take(delta_pose, train_indices, axis=0)
    #
    # Split training data into training + validation set

    val_samples = 0
    # val_pose_pre = train_pose_pre[:val_samples]
    # val_img_pre = train_img_pre[:val_samples]
    # val_actions = train_actions[:val_samples]
    # val_pose_post = train_pose_post[:val_samples]
    # val_img_post = train_img_post[:val_samples]
    # val_pose_delta = train_pose_delta[:val_samples]

    train_pose_pre = train_pose_pre[val_samples:]
    train_img_pre = train_img_pre[val_samples:]
    train_actions = train_actions[val_samples:]
    train_pose_post = train_pose_post[val_samples:]
    train_img_post = train_img_post[val_samples:]
    train_pose_delta = train_pose_delta[val_samples:]

    test_pose_pre = np.take(pre_pose, test_indices, axis=0)
    test_img_pre = np.take(pre_img, test_indices, axis=0)
    test_actions = np.take(actions, test_indices, axis=0)
    test_pose_post = np.take(post_pose, test_indices, axis=0)
    test_img_post = np.take(post_img, test_indices, axis=0)
    test_pose_delta = np.take(delta_pose, test_indices, axis=0)

    np.save('../train_data/{}-task-states-pose.npy'.format(task), train_pose_pre)
    np.save('../train_data/{}-task-states-img.npy'.format(task), train_img_pre)
    np.save('../train_data/{}-task-actions.npy'.format(task), train_actions)
    np.save('../train_data/{}-task-effects-pose.npy'.format(task), train_pose_post)
    np.save('../train_data/{}-task-effects-img.npy'.format(task), train_img_post)
    np.save('../train_data/{}-task-effects-delta.npy'.format(task), train_pose_delta)

    # np.save('../val_data/{}-task-states-pose.npy'.format(task), val_pose_pre)
    # np.save('../val_data/{}-task-states-img.npy'.format(task), val_img_pre)
    # np.save('../val_data/{}-task-actions.npy'.format(task), val_actions)
    # np.save('../val_data/{}-task-effects-pose.npy'.format(task), val_pose_post)
    # np.save('../val_data/{}-task-effects-img.npy'.format(task), val_img_post)
    # np.save('../val_data/{}-task-effects-delta.npy'.format(task), val_pose_delta)

    np.save('../test_data/{}-task-states-pose.npy'.format(task), test_pose_pre)
    np.save('../test_data/{}-task-states-img.npy'.format(task), test_img_pre)
    np.save('../test_data/{}-task-actions.npy'.format(task), test_actions)
    np.save('../test_data/{}-task-effects-pose.npy'.format(task), test_pose_post)
    np.save('../test_data/{}-task-effects-img.npy'.format(task), test_img_post)
    np.save('../test_data/{}-task-effects-delta.npy'.format(task), test_pose_delta)


def scale_data(task):
    # Scale train, test
    state_scaler = MinMaxScaler()
    effect_scaler = MinMaxScaler()
    effect_scaler_delta = MinMaxScaler()

    train_state_pose = np.load('../train_data/{}-task-states-pose.npy'.format(task), allow_pickle=True)
    train_effect_pose = np.load('../train_data/{}-task-effects-pose.npy'.format(task), allow_pickle=True)
    train_delta_pose = np.load('../train_data/{}-task-effects-delta.npy'.format(task), allow_pickle=True)

    # val_state_pose = np.load('../val_data/{}-task-states-pose.npy'.format(task), allow_pickle=True)
    # val_effect_pose = np.load('../val_data/{}-task-effects-pose.npy'.format(task), allow_pickle=True)
    # val_delta_pose = np.load('../val_data/{}-task-effects-delta.npy'.format(task), allow_pickle=True)

    test_state_pose = np.load('../test_data/{}-task-states-pose.npy'.format(task), allow_pickle=True)
    test_effect_pose = np.load('../test_data/{}-task-effects-pose.npy'.format(task), allow_pickle=True)
    test_delta_pose = np.load('../test_data/{}-task-effects-delta.npy'.format(task), allow_pickle=True)

    # input scaling
    train_state_pose_scaled = state_scaler.fit_transform(train_state_pose)
    # val_state_pose_scaled = state_scaler.transform(val_state_pose)
    test_state_pose_scaled = state_scaler.transform(test_state_pose)

    # option 1: effect pose scaling
    train_effect_pose_scaled = effect_scaler.fit_transform(train_effect_pose)
    # val_effect_pose_scaled = effect_scaler.transform(val_effect_pose)
    test_effect_pose_scaled = effect_scaler.transform(test_effect_pose)

    # option 2: effect delta scaling
    train_effect_delta_scaled = effect_scaler_delta.fit_transform(train_delta_pose)
    # val_effect_delta_scaled = effect_scaler_delta.transform(val_delta_pose)
    test_effect_delta_scaled = effect_scaler_delta.transform(test_delta_pose)

    np.save('../train_data/{}-task-states-pose-scaled.npy'.format(task), train_state_pose_scaled)
    # np.save('../val_data/{}-task-states-pose-scaled.npy'.format(task), val_state_pose_scaled)
    np.save('../test_data/{}-task-states-pose-scaled.npy'.format(task), test_state_pose_scaled)

    np.save('../train_data/{}-task-effects-pose-scaled.npy'.format(task), train_effect_pose_scaled)
    # np.save('../val_data/{}-task-effects-pose-scaled.npy'.format(task), val_effect_pose_scaled)
    np.save('../test_data/{}-task-effects-pose-scaled.npy'.format(task), test_effect_pose_scaled)

    np.save('../train_data/{}-task-effects-delta-scaled.npy'.format(task), train_effect_delta_scaled)
    # np.save('../val_data/{}-task-effects-delta-scaled.npy'.format(task), val_effect_delta_scaled)
    np.save('../test_data/{}-task-effects-delta-scaled.npy'.format(task), test_effect_delta_scaled)


if __name__ == '__main__':
    task_names = ["hit", "push", "stack"]
    for task_name in task_names:
        # if task_name == "stack":
        #     train_test_split(task_name, samples=500)
        # else:
        #     train_test_split(task_name)
        scale_data(task_name)
