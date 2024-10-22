import argparse
import os
import time
from multiprocessing import Process
from task import TableTopTask

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Collect action-effect data for a task")
    parser.add_argument("-d", help="data folder (default:data/)", type=str, default="simulation_data/")
    parser.add_argument("-N", help="number of samples per task(default:500)", default=[72])
    parser.add_argument("-t", help="list of target simulation files",  # target .py files
                        default=[TableTopTask])
    args = parser.parse_args()

    if not os.path.exists(args.d):
        os.makedirs(args.d)

    procs = []
    start = time.time()
    task_names = ['stack']

    for i, target in enumerate(args.t):
        p = Process(target=target, args=[args.d, str(args.N[i]), task_names[i]])
        p.start()
        procs.append(p)

    for i in range(len(args.t)):
        procs[i].join()

    end = time.time()
    elapsed = end - start
    # print(f"Collected {len(args.t) * args.N} samples in {elapsed:.2f} seconds. {len(args.t) * args.N / elapsed}")
