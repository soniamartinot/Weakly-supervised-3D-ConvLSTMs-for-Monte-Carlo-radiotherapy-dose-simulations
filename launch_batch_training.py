from multiprocessing import Pool
import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("n_pools")
args = parser.parse_args()

n_pools = int(args.n_pools)

def launching(n_pool):
    os.system("python3 launch_trainings.py  {} {}".format(n_pools, n_pool))

    
pool = Pool(n_pools)
pool.map(launching, np.arange(1, n_pools + 1).tolist())
