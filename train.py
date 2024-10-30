import argparse
import os
import time
from multiprocessing import Process, Lock

import numpy as np
import torch
import yaml

from network.models import SingleTask, MultiTask


def train(log_lock, seed, config):
    log_lock.acquire()
    try:
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
        print(config)
    finally:
        log_lock.release()
    if config["mode"] == "singletask":
        model = SingleTask(config)
    elif config["mode"] == "multitask":
        model = MultiTask(config)
    else:
        raise ValueError("Invalid model.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Interleaved multi effect prediction models.")
    parser.add_argument("-opts", help="option file", type=str, required=True,
                        default='opts.yml')
    args = parser.parse_args()

    opts = yaml.safe_load(open(args.opts, "r"))
    if not os.path.exists(opts["save"]):
        os.makedirs(opts["save"])

    train_opts = opts.copy()
    opts["time"] = time.asctime(time.localtime(time.time()))
    seeds = np.random.randint(low=0, high=100000, size=opts["num_seeds"])
    opts["seeds"] = seeds

    # Save training config
    file = open(os.path.join(opts["save"], "opts.yaml"), "w")
    yaml.dump(opts, file)
    file.close()
    print(yaml.dump(opts))

    lock = Lock()

    procs = []
    for i in range(train_opts["num_seeds"]):
        p = Process(target=train, args=(lock, str(seeds[i]), train_opts))
        p.start()
        procs.append(p)

    for i in range(train_opts["num_seeds"]):
        procs[i].join()
