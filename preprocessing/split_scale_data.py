import numpy as np
from sklearn.preprocessing import MinMaxScaler


def train_test_split(task):
    task_name = task
    pre_pose = np.load('../data/{}-task-states-pose.npy'.format(task_name), allow_pickle=True)
    pre_img = np.load('../data/{}-task-states-img.npy'.format(task_name), allow_pickle=True)
    actions = np.load('../data/{}-task-actions.npy'.format(task_name), allow_pickle=True)
    post_pose = np.load('../data/{}-task-effects-pose.npy'.format(task_name), allow_pickle=True)
    post_img = np.load('../data/{}-task-effects-img.npy'.format(task_name), allow_pickle=True)

    # train_indices = np.random.choice(a=range(5000), size=4500, replace=False)
    # test_indices = np.setdiff1d(range(5000), train_indices, assume_unique=True)
    #

    train_indices = np.random.choice(a=range(5000), size=4000, replace=False)
    test_indices = np.setdiff1d(range(5000), train_indices, assume_unique=True)

    train_pose_pre = np.take(pre_pose, train_indices, axis=0)
    train_img_pre = np.take(pre_img, train_indices, axis=0)
    train_actions = np.take(actions, train_indices, axis=0)
    train_pose_post = np.take(post_pose, train_indices, axis=0)
    train_img_post = np.take(post_img, train_indices, axis=0)
    #
    # # Split training data into training + validation set

    # val_pose_pre = train_pose_pre[:500]
    # val_img_pre = train_img_pre[:500]
    # val_actions = train_actions[:500]
    # val_pose_post = train_pose_post[:500]
    # val_img_post = train_img_post[:500]

    # train_pose_pre = train_pose_pre[500:]
    # train_img_pre = train_img_pre[500:]
    # train_actions = train_actions[500:]
    # train_pose_post = train_pose_post[500:]
    # train_img_post = train_img_post[500:]

    test_pose_pre = np.take(pre_pose, test_indices, axis=0)
    test_img_pre = np.take(pre_img, test_indices, axis=0)
    test_actions = np.take(actions, test_indices, axis=0)
    test_pose_post = np.take(post_pose, test_indices, axis=0)
    test_img_post = np.take(post_img, test_indices, axis=0)

    np.save('../train_data/{}-task-states-pose.npy'.format(task_name), train_pose_pre)
    np.save('../train_data/{}-task-states-img.npy'.format(task_name), train_img_pre)
    np.save('../train_data/{}-task-actions.npy'.format(task_name), train_actions)
    np.save('../train_data/{}-task-effects-pose.npy'.format(task_name), train_pose_post)
    np.save('../train_data/{}-task-effects-img.npy'.format(task_name), train_img_post)

    # np.save('val_data/{}-task-states-pose.npy'.format(task_name), val_pose_pre)
    # np.save('val_data/{}-task-states-img.npy'.format(task_name), val_img_pre)
    # np.save('val_data/{}-task-actions.npy'.format(task_name), val_actions)
    # np.save('val_data/{}-task-effects-pose.npy'.format(task_name), val_pose_post)
    # np.save('val_data/{}-task-effects-img.npy'.format(task_name), val_img_post)

    np.save('../test_data/{}-task-states-pose.npy'.format(task_name), test_pose_pre)
    np.save('../test_data/{}-task-states-img.npy'.format(task_name), test_img_pre)
    np.save('../test_data/{}-task-actions.npy'.format(task_name), test_actions)
    np.save('../test_data/{}-task-effects-pose.npy'.format(task_name), test_pose_post)
    np.save('../test_data/{}-task-effects-img.npy'.format(task_name), test_img_post)


def scale_data(task):
    # Scale train, val, test
    task_name = task
    pose_scaler = MinMaxScaler()
    # action_scaler = MinMaxScaler()

    train_pose = np.load('../train_data/{}-task-states-pose.npy'.format(task_name), allow_pickle=True)
    # train_actions = np.load('train_data/{}-task-actions.npy'.format(task_name), allow_pickle=True)

    # val_pose = np.load('val_data/{}-task-states-pose.npy'.format(task_name), allow_pickle=True)
    # val_actions = np.load('val_data/{}-task-actions.npy'.format(task_name), allow_pickle=True)

    test_pose = np.load('../test_data/{}-task-states-pose.npy'.format(task_name), allow_pickle=True)
    # test_actions = np.load('test_data/{}-task-actions.npy'.format(task_name), allow_pickle=True)

    scaled_pose_train = pose_scaler.fit_transform(train_pose)
    # scaled_pose_val = pose_scaler.transform(val_pose)
    scaled_pose_test = pose_scaler.transform(test_pose)

    # scaled_action_train = action_scaler.fit_transform(train_actions)
    # scaled_action_val = action_scaler.transform(val_actions)
    # scaled_action_test = action_scaler.transform(test_actions)

    np.save('../train_data/{}-task-states-pose-scaled.npy'.format(task_name), scaled_pose_train)
    # np.save('train_data/{}-task-actions-scaled.npy'.format(task_name), scaled_action_train)
    # np.save('val_data/{}-task-states-pose-scaled.npy'.format(task_name), scaled_pose_val)
    # np.save('val_data/{}-task-actions-scaled.npy'.format(task_name), scaled_action_val)
    np.save('../test_data/{}-task-states-pose-scaled.npy'.format(task_name), scaled_pose_test)
    # np.save('test_data/{}-task-actions-scaled.npy'.format(task_name), scaled_action_test)


if __name__ == '__main__':
    scale_data("hit")
