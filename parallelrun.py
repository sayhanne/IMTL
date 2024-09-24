import argparse
from multiprocessing import Process, Lock
from run import Runner
import numpy as np

if __name__ == '__main__':
    epoch = 1000
    save_path = 'results/'

    parser = argparse.ArgumentParser("Multi Task Training")
    parser.add_argument("--seq-path", help="Task sequence file",
                        default=None)
    parser.add_argument("-save-path", help="Save results path",
                        default=save_path)
    parser.add_argument("-t", help="Target run file",  # target .py file
                        default=Runner)
    args = parser.parse_args()
    models = ['b1', 'b2', 'lp', 'lpe']
    sel = ['rand', 'rand', 'lp', 'lpe']
    share_bb = [1, 1, 1, 1]
    procs = []
    num_seeds = 1
    seeds = np.random.randint(low=0, high=100000, size=num_seeds)
    print(seeds)

    for index, model in enumerate(models):
        lock = Lock()
        for i in range(num_seeds):
            p = Process(target=args.t, args=[lock, str(seeds[i]), model, sel[index], str(share_bb[index]),
                                             str(epoch), args.seq_path, args.save_path])
            p.start()
            procs.append(p)

        for i in range(num_seeds):
            procs[i].join()

        procs = []
        print('--------------Model {} run finished!-----------------'.format(model))
