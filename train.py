import argparse
import os
import time
from copy import deepcopy
from multiprocessing import Process, Lock

import numpy as np
import torch
import yaml

from network.models import SingleTask, MultiTask, BlockedMultiTask
from preprocessing.dataset import LocationPredictionDataset


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

    for task_name in config["tasks"]:
        train_dataset = LocationPredictionDataset(task_name=task_name, batch_size=config["batch_size"], mode="train")
        train_loader = train_dataset.load_data()
        train_loaders.append(train_loader)

        val_dataset = LocationPredictionDataset(task_name=task_name, batch_size=config["batch_size"], mode="test")
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
    model.train_(train_loaders, val_loaders)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Interleaved multi effect prediction models.")
    parser.add_argument("-opts", help="option file", type=str,
                        default='singletask.yml')
    args = parser.parse_args()

    opts = yaml.safe_load(open(args.opts, "r"))
    if not os.path.exists(opts["save"]):
        os.makedirs(opts["save"])

    train_opts = deepcopy(opts)
    opts["time"] = time.asctime(time.localtime(time.time()))
    # seeds = np.random.randint(low=0, high=100000, size=opts["num_seeds"])
    # print(seeds)
    seeds = np.asarray([66094,  8571, 65138, 61881, 85675, 29433, 46911, 51577, 92058, 36322])
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
