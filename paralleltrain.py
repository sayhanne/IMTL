import argparse
import os
import time
from multiprocessing import Process, Lock

import numpy as np
import torch
import yaml


# TODO: fix
def train(log_lock, seed, config):
    log_lock.acquire()
    print("Model name", config["name"])
    print("Seed", seed)
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    try:
        if config["model"] == "mtl":
            model = MTL(config)
        elif config["model"] == "vanillamlp":
            model = MLP(config)
        elif config["model"] == "imtl":
            model = IMTL(config)
        elif config["model"] == "blocked":
            model = MTL(config)
        else:
            raise ValueError("Invalid model.")
    finally:
        log_lock.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Interleaved multi effect prediction models.")
    parser.add_argument("-opts", help="option file", type=str, required=True,
                        default='opts.yml')
    args = parser.parse_args()

    opts = yaml.safe_load(open(args.opts, "r"))
    if not os.path.exists(opts["save"]):
        os.makedirs(opts["save"])
    opts["time"] = time.asctime(time.localtime(time.time()))
    seeds = np.random.randint(low=0, high=100000, size=opts["num_seeds"])
    device = torch.device(opts["device"])
    lock = Lock()

    procs = []
    for i in range(opts["num_seeds"]):
        p = Process(target=train, args=(lock, str(seeds[i]), opts))
        p.start()
        procs.append(p)

    for i in range(opts["num_seeds"]):
        procs[i].join()

    procs = []

    # Save training config
    file = open(os.path.join(opts["save"], "opts.yaml"), "w")
    yaml.dump(opts, file)
    file.close()
    print(yaml.dump(opts))
