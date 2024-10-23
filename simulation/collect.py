import argparse
import os
import time
from multiprocessing import Process
from task import TableTopTask

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Collect action-effect data for a task")
    parser.add_argument("-d", help="data folder (default:data/)", type=str, default="simulation_data/")
    parser.add_argument("-N", help="number of samples per task(default:500)", default=[9000, 9000, 9000])
    parser.add_argument("-t", help="target simulation file",  # target .py file
                        default=TableTopTask)
    args = parser.parse_args()

    if not os.path.exists(args.d):
        os.makedirs(args.d)

    procs = []
    start = time.time()
    task_names = ['push', 'hit', 'stack']

    for i, task_name in enumerate(task_names):
        p = Process(target=args.t, args=[args.d, str(args.N[i]), task_name])
        p.start()
        procs.append(p)

    for i in range(len(task_names)):
        procs[i].join()

    end = time.time()
    elapsed = end - start
    # print(f"Collected {len(args.t) * args.N} samples in {elapsed:.2f} seconds. {len(args.t) * args.N / elapsed}")
