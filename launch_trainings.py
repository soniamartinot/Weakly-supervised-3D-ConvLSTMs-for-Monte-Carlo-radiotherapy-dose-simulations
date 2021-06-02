import os
import argparse
import numpy as np
from itertools import product
parser = argparse.ArgumentParser()
parser.add_argument("n_pools")
parser.add_argument('n_pool')
args = parser.parse_args()

n_pools = int(args.n_pools)
n_pool = int(args.n_pool)

# Paths
BASE_PATH = "/workspace/workstation/"
save_path = BASE_PATH + "lstm_denoising/noise2noise/"


# Hyperparameters
# models = ["stack3D_deep"]
models = ["stack3D_deep_activation"]
# models = ["stack3D_filters"]
# models = ["stack3D"]
# models = ["lunet4-bn-leaky3D"]
# models = ["lunet4-bn-leaky3D_big"]
# models = ["MCDNet3D"]
# models = ["stack3D_deep"]
# models = ["bionet3d"]
# models = ["unet3d"]
num_layers = 3



kernel_size = 3
n_samples = 30000
patch_sizes = [32]
depth = 32


all_channels = False
normalized = [False]
standardize = False
raw = False


uncertainty_thresh = 0.3
dose_thresh = 0.1

n_frames = 3
batch_size = 6
optimizers = ['adam']
# optimizers = ["sgd"]
weight_decays = [3e-4]
learning_rates = [1e-3]

high_dose_only = False
p1 = 0.6
p2 = 0.01

# Loss
# loss_names = ["ssim-smoothl1"]
loss_names = ['ssim-mse']
# loss_names = ["ssim"]
# loss_names = ["mse"]
# loss_names = ["l1smooth"]
# loss_names = ["l1"]
# loss_names = ["ssim", "ssim-smoothl1", "mse"]

# CT related parameters: if we minimax the images
cts = [False]
single_frames = [False, True]
# single_frames = [True]
single_frames = [False]

select_anatomy = False
anatomy = "orl"

# mode = "finite"
mode = "infinite"

lr_scheduler = "cosine"
# lr_scheduler = "plateau"

noise2noise = True

last_activations = ["leakyrelu"]

LAMBDA = 100
if models[0] != "pix2pix3d":    LAMBDA = None

all_param = list(product(models, single_frames, last_activations, cts, patch_sizes, normalized, optimizers, loss_names, learning_rates, weight_decays))


command = "python3 train_infinite.py {n_samples} {patch_size} {model_name} {loss_name} {learning_rate} {weight_decay} {optimizer_name} {save_path} {gpu_number} {all_channels} {normalized_by_gt} {standardize} {uncertainty_thresh} {dose_thresh} {n_frames} {batch_size} {add_ct} {ct_norm} {high_dose_only} {p1} {p2} {single_frame} {select_anatomy} {anatomy} {mode} {lr_scheduler} {depth} {raw} {kernel_size} {noise2noise} {num_layers} {last_activation} {LAMBDA}"


        
        
def create_commands():
    all_commands = []
    fraction = int(np.ceil(len(all_param) / n_pools))
    for n in range(1, n_pools+1):
        if n == n_pool:
            for i in range(fraction*(n-1), fraction*n):                
                if i < len(all_param):
                    model_name, single_frame, last_activation, ct, patch_size, normalized_by_gt, optimizer_name, loss_name, learning_rate, weight_decay = all_param[i]
                    add_ct = ct
                    ct_norm = ct
                    for n in range(1, n_pools+1):
                        a, b = fraction*(n-1), fraction*n
                        if a <= i < b: gpu_number = n
                        gpu_number = 4
#                         gpu_number = 7

                    print("Launching {}: model: {}   lr: {}   loss: {}    optimizer: {}  gpu_number: {}".format(i, 
                                                                                                                model_name, 
                                                                                                                learning_rate, 
                                                                                                                loss_name, 
                                                                                                                optimizer_name,
                                                                                                                gpu_number))
                    to_command = command.format(n_samples=n_samples,
                                                patch_size=patch_size,
                                                model_name=model_name,
                                                loss_name=loss_name,
                                                learning_rate=learning_rate,
                                                weight_decay=weight_decay,
                                                optimizer_name=optimizer_name,
                                                save_path=save_path,
                                                gpu_number=gpu_number,
                                                all_channels=all_channels,
                                                normalized_by_gt=normalized_by_gt,
                                                standardize=standardize,
                                                dose_thresh=dose_thresh,
                                                uncertainty_thresh=uncertainty_thresh,
                                                n_frames=n_frames,
                                                batch_size=batch_size,
                                                add_ct=add_ct,
                                                ct_norm=ct_norm,
                                                high_dose_only=high_dose_only,
                                                p1=p1,
                                                p2=p2,
                                                single_frame=single_frame,
                                                select_anatomy=select_anatomy,
                                                anatomy=anatomy,
                                                mode=mode,
                                                lr_scheduler=lr_scheduler,
                                                depth=depth,
                                                raw=raw,
                                                kernel_size=kernel_size,
                                                noise2noise=noise2noise,
                                                num_layers=num_layers,
                                                last_activation=last_activation,
                                                LAMBDA=LAMBDA)

                    all_commands += [to_command]
        
        
    return all_commands

all_commands = create_commands()
# for c in all_commands: print("\n", c)
    
    
def launch(i):
        os.system(all_commands[i])



fraction = int(np.ceil(len(all_param) / n_pools))
print("Total number of trainings with n_pools={} : {} {} {}".format(n_pools, len(all_param), len(all_param) / n_pools, fraction))
for n in range(1, n_pools+1):
    if n == n_pool:
        for i in range(fraction*(n-1), fraction*n):                launch(i)
            


        
