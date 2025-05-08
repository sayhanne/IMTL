import argparse
import os
import time
from copy import deepcopy
from multiprocessing import Process, Lock

import numpy as np
import torch
import yaml

from network.models import SingleTask, MultiTask, BlockedMultiTask
from preprocessing.dataset import EffectPredictionDataset  # default_transform


def get_parameter_count(model):
    total_num = 0
    for param in model.parameters():
        total_num += param.shape.numel()
    return total_num


def train(log_lock, seed, config):
    log_lock.acquire()
    try:
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
    finally:
        log_lock.release()

    # Train set
    train_loaders = []
    val_loaders = []
    # ext = "img" if "cnn" in config else "pose-scaled"  # img input or not
    ext = "pose-scaled"
    for task_name in config["tasks"]:
        train_dataset = EffectPredictionDataset(task_name=task_name, ext_=ext, batch_size=config["batch_size"],
                                                mode="train", y=config["target"])
        train_loader = train_dataset.load_data()
        train_loaders.append(train_loader)
        val_dataset = EffectPredictionDataset(task_name=task_name, ext_=ext, batch_size=config["batch_size"],
                                              mode="test", y=config["target"])
        val_loader = val_dataset.load_data()
        val_loaders.append(val_loader)

    if config["mode"] == "singletask":
        model = SingleTask(seed, config)
    elif config["mode"] == "multitask":
        model = MultiTask(seed, config)
    elif config["mode"] == "blocked":
        model = BlockedMultiTask(seed, config)
    else:
        raise ValueError("Invalid model.")
    # print(model)
    # print("parameter count", get_parameter_count(model))
    model.train_(train_loaders, val_loaders)


def object_based_transfer(seed, config):
    task_name_id = {"push": 0, "hit": 1, "stack": 2}
    object_dict = {"sphere": [1, 0, 0, 0, 0, 0],
                   "cube": [0, 1, 0, 0, 0, 0],
                   "ver-cylinder": [0, 0, 1, 0, 0, 0],
                   "hor-cylinder": [0, 0, 0, 1, 0, 0],
                   "ver-prism": [0, 0, 0, 0, 1, 0],
                   "hor-prism": [0, 0, 0, 0, 0, 1]}
    ext = "pose-scaled"
    model = MultiTask(seed, config)
    model.load(path="results/training-save/imtl-lp/model_ckpts", ext="_last")
    per_object_results = {"sphere": {}, "cube": {},
                          "ver-cylinder": {}, "hor-cylinder": {},
                          "ver-prism": {}, "hor-prism": {}}

    for target_obj_type, t_idx in object_dict.items():
        for object_type, idx in object_dict.items():
            subset = EffectPredictionDataset(task_name="stack", ext_=ext,
                                             y=config["target"], object_id=idx, target_id=t_idx)
            delta = model.evaluate_single_task_contribution(subset.load_data(), active_task_id=task_name_id["stack"])
            per_object_results[target_obj_type][object_type] = {ot_name: delta[ot_id] for ot_name, ot_id in
                                                                task_name_id.items()
                                                                if ot_id in delta}
    np.save("seed-{}-transfer-two-obj.npy".format(seed), per_object_results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("iMTLIFE")
    parser.add_argument("-opts", help="option file", type=str,
                        default='multitask.yml')
    args = parser.parse_args()

    opts = yaml.safe_load(open(args.opts, "r"))
    if not os.path.exists(opts["save"]):
        os.makedirs(opts["save"])

    # Create subdir for model checkpoints and results
    if not os.path.exists(opts["save"] + "/model_ckpts"):
        os.makedirs(opts["save"] + "/model_ckpts")
    if not os.path.exists(opts["save"] + "/plots"):
        os.makedirs(opts["save"] + "/plots")

    train_opts = deepcopy(opts)
    opts["time"] = time.asctime(time.localtime(time.time()))
    # seeds = np.random.randint(low=0, high=10000, size=opts["num_seeds"])
    # print(seeds)
    seeds = np.asarray([8302, 2766, 257, 7600, 6657, 8226, 6841, 4908, 1321, 7857])
    opts["seeds"] = seeds.tolist()

    # Save training config
    file = open(os.path.join(opts["save"], "opts.yaml"), "w")
    yaml.dump(opts, file)
    file.close()
    print(yaml.dump(opts))

    lock = Lock()
    procs = []
    for i in range(train_opts["num_seeds"]):
        p = Process(target=train, args=(lock, seeds[i], train_opts))
        p.start()
        procs.append(p)

    for i in range(train_opts["num_seeds"]):
        procs[i].join()
