import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from platform import python_version
from datetime import datetime
import numpy as np
from glob import glob
from tqdm import tqdm
import os, sys, time, random
import SimpleITK as sitk
from matplotlib.backends.backend_pdf import PdfPages

import pytorch_ssim

BASE_PATH = "/workspace/workstation/"

sys.path.append(BASE_PATH + 'lstm_denoising')
from models.just_imports import *
from models.imports3D import *



# This is a trainer script.
# Inputs should give :
#      - the number of samples required in the training set
#      - The model used should be also given by its proper name
#      - the desired loss
#      - the learning rate
#      - the weight decay
#      - the optimizer

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("n_samples")
parser.add_argument("patch_size")
parser.add_argument("model_name")
parser.add_argument("loss_name")
parser.add_argument("learning_rate")
parser.add_argument("weight_decay")
parser.add_argument("optimizer_name")
parser.add_argument("save_path")
parser.add_argument('gpu_number')
parser.add_argument('all_channels')
parser.add_argument('normalized_by_gt')
parser.add_argument("standardize")
parser.add_argument("uncertainty_thresh")
parser.add_argument("dose_thresh")
parser.add_argument("n_frames")
parser.add_argument("batch_size")
parser.add_argument("add_ct")
parser.add_argument("ct_norm")
parser.add_argument("high_dose_only")
parser.add_argument("p1")
parser.add_argument("p2")
parser.add_argument("single_frame")
parser.add_argument("select_anatomy")
parser.add_argument("anatomy")
parser.add_argument("mode")
parser.add_argument("lr_scheduler")
parser.add_argument("depth")
parser.add_argument("raw")
parser.add_argument("kernel_size")
parser.add_argument("noise2noise")
parser.add_argument("num_layers")
parser.add_argument("last_activation")
parser.add_argument("LAMBDA")
args = parser.parse_args()


# Set gpu
gpu_number = int(args.gpu_number)
torch.cuda.set_device(gpu_number)
print("Current GPU:", torch.cuda.current_device())



def list_cases(path, exclude=[]):
    return [p for p in glob(path + "*") if len(os.path.basename(p)) == 4 and not os.path.basename(p) in exclude]

def ssim_smoothl1(out, gt):
    return 20*nn.SmoothL1Loss()(out, gt) - pytorch_ssim.SSIM()(out, gt)

def ssim_mse(out, gt):
    return 20*nn.MSELoss()(out, gt) - pytorch_ssim.SSIM()(out, gt)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
    
    
# Hyperparameters
n_samples =   int(args.n_samples)
batch_size =  int(args.batch_size)
patch_size =  int(args.patch_size)
n_frames =    int(args.n_frames)
depth =       int(args.depth)
kernel_size = int(args.kernel_size)
num_layers =  int(args.num_layers)


uncertainty_thresh = float(args.uncertainty_thresh)
dose_thresh =        float(args.dose_thresh)
learning_rate =      float(args.learning_rate)
weight_decay =       float(args.weight_decay)
p1 =                 float(args.p1)
p2 =                 float(args.p2)

print("Batch size: {}".format(batch_size))

all_channels =     args.all_channels == 'True'
normalized_by_gt = args.normalized_by_gt == "True"
standardize =      args.standardize == "True"
raw =              args.raw == "True"
add_ct =           args.add_ct == "True"
ct_norm =          args.ct_norm == "True"
select_anatomy =   args.select_anatomy == "True"
high_dose_only =   args.high_dose_only == 'True'
single_frame =     args.single_frame == "True"
noise2noise =      args.noise2noise == 'True'

model_name = args.model_name
if "bionet" in model_name or "unet3d" in model_name: num_layers =  int(args.num_layers)
else: num_layers = None

anatomy = args.anatomy
mode = args.mode
lr_scheduler = args.lr_scheduler

if "activation" in model_name: last_activation = args.last_activation
else: last_activation = "tanh"


# Load the dataset
if select_anatomy:
    anat_dict = np.load("/workspace/workstation/dataset_anatomy.npy", allow_pickle=True).item()
    cases = anat_dict[anatomy]
    n_train = 20
    n_validation = 5
else:
    simu_path = BASE_PATH + "lstm_denoising/numpy_simulations/"
    cases = list_cases(simu_path, exclude=['0112'])
    n_train = 40
    n_validation = 5

# Is the model a UNet architecture ?
if model_name[0] == 'u' or "MCD" in model_name or "bio" in model_name: unet=True
else: unet=False
    
n_val = 128
val_batch_size = 4

train = MC3D_Dataset_Noise2Noise(cases[:n_train], n_frames=n_frames, patch_size=patch_size, 
                       all_channels=all_channels, normalized_by_gt=normalized_by_gt, standardize=standardize,
                       uncertainty_thresh=uncertainty_thresh, dose_thresh=dose_thresh,
                       unet=unet, add_ct=add_ct, ct_norm=ct_norm,
                       high_dose_only=high_dose_only, p1=p1, p2 = p2,
                       single_frame=single_frame, 
                       mode=mode, n_samples=n_samples, 
                       depth=depth, raw=raw,
                       noise2noise=noise2noise)
val = MC3D_Dataset_Noise2Noise(cases[n_train:n_train+n_validation], n_frames=n_frames, patch_size=patch_size,
                       all_channels=all_channels, normalized_by_gt=normalized_by_gt, standardize=standardize,
                       uncertainty_thresh=uncertainty_thresh, dose_thresh=dose_thresh,
                       unet=unet, add_ct=add_ct, ct_norm=ct_norm,
                       high_dose_only=high_dose_only, p1=0.5, p2=0.1,
                       single_frame=single_frame,
                       mode="finite", n_samples=128,
                       depth=depth, raw=raw,
                       noise2noise=False)


train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=6)
val_loader = DataLoader(val, batch_size=val_batch_size, shuffle=False, num_workers=6)

print("Number of training batches: \t{}".format(len(train_loader)))
print("Number of validation batches: \t{}".format(n_val))

# Load model
if model_name == "stack3D":                    model = stack_model_3D()
elif model_name == "stack3D_filters":          model = stack_3D(kernel_size=kernel_size)
elif model_name == "stack3D_activation":       model = stack_model_3D_activation(last_activation=last_activation)
elif model_name == "stack3D_deep_activation":  model = stack_model_3D_deep_activation(last_activation=last_activation)
elif model_name == "stack3D_deep":             model = stack_model_3D_deep()
elif model_name == "lunet4-bn-leaky3D":  
    model = LUNet4BNLeaky3D()
    unet = False
elif model_name == "lunet4-bn-leaky3D_big":  
    model = LUNet4BNLeaky3D_big()
    unet = False
elif model_name == "MCDNet3D":             
    if single_frame and add_ct:       model = MCDNet3D(2)
    elif single_frame :               model = MCDNet3D(1)
    elif not single_frame and add_ct: model = MCDNet3D(n_frames +1)
    else:                             model = MCDNet3D(n_frames)
elif model_name == "bionet3d": 
    if single_frame and add_ct:       model = BiONet(input_channels=2, num_layers=num_layers)
    elif single_frame:                model = BiONet(input_channels=1, num_layers=num_layers)
    elif not single_frame and add_ct: model = BiONet(input_channels=n_frames + 1, num_layers=num_layers)
    else:                             model = BiONet(input_channels=n_frames, num_layers=num_layers)
elif model_name == "unet3d":
    if single_frame and add_ct:       model = UNet3D(n_frames=2, num_layers=num_layers)
    elif single_frame:                model = UNet3D(n_frames=1, num_layers=num_layers)
    elif not single_frame and add_ct: model = UNet3D(n_frames=n_frames + 1, num_layers=num_layers)
    else:                             model = UNet3D(n_frames=n_frames, num_layers=num_layers)
else: 
    print("Unrecognized model")
    for i in range(1): break
print("Model: ", model_name)
model.cuda()

    
# Set up the loss: to replace with parameters to change
loss_name = args.loss_name
if loss_name == "mse": loss = nn.MSELoss()
elif loss_name == "l1": loss = nn.L1Loss()
elif loss_name == "l1smooth": loss = nn.SmoothL1Loss()
elif loss_name == "ssim": loss = pytorch_ssim.SSIM()
elif loss_name == "ssim-smoothl1": loss = ssim_smoothl1
elif loss_name == "ssim-mse": loss = ssim_mse
print("Loss used: ", loss_name)

# Set up the optimizer
optimizer_name = args.optimizer_name
params = [p for p in model.parameters() if p.requires_grad]
if optimizer_name == "adam": optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
elif optimizer_name == "adamw": optimizer = torch.optim.AdamW(params, lr=learning_rate)
elif optimizer_name == "sgd": optimizer = torch.optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)
    
# Don't forget to save all intermediate results and model
save_path = args.save_path 
now = str(datetime.now())
training_name = "train_{}_{}".format(now.split(' ')[0], now.split(' ')[-1])

save_path = save_path + '/trainings'

os.system("mkdir {save_path}/{training_name}".format(save_path=save_path,
                                                    training_name=training_name))
save_path = save_path + "/" + training_name
print(training_name)


# Write configuration file
from configparser import ConfigParser
config_object = ConfigParser()

config_object["MODEL"] = {"model_name":       model_name,
                          "loss_name":        loss_name,
                          "optimizer_name":   optimizer_name,
                          "learning_rate":    learning_rate,
                          "weight_decay":     weight_decay,
                          "gpu_number":       gpu_number,
                          "lr_scheduler":     lr_scheduler,
                          "kernel_size":      kernel_size,
                          "num_layers":       str(num_layers),
                          "last_activation":  str(last_activation)}

config_object["DATASETINFO"] = {"all_channels":      all_channels,
                                "normalized_by_gt":  normalized_by_gt,
                                "standardize":       standardize,
                                "n_samples":         n_samples,
                                "mode":              mode, 
                                "n_frames":          n_frames,
                                "patch_size":        patch_size,
                                "depth":             depth,
                                "batch_size":        batch_size,
                                "uncertainty_thresh":uncertainty_thresh,
                                "dose_thresh":       dose_thresh, 
                                "add_ct":            add_ct,
                                "ct_norm":           ct_norm,
                                "high_dose_only":    high_dose_only,
                                "p1":                p1,
                                "p2":                p2,
                                "single_frame":      single_frame,
                                "select_anatomy":    select_anatomy,
                                "anatomy":           anatomy, 
                                "raw":               raw} 

with open(save_path + '/config.ini', 'w') as conf:
    config_object.write(conf)


# Learning rate decay
if lr_scheduler == "plateau":
    decayRate = 0.8
    my_lr_scheduler= torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                                mode='min', 
                                                                factor=decayRate, 
                                                                patience=15, 
                                                                threshold=1e-2, 
                                                                threshold_mode='rel', 
                                                                cooldown=5, 
                                                                min_lr=1e-7, 
                                                                eps=1e-08, 
                                                                verbose=True)
elif lr_scheduler == "cosine":
# Learning rate update: cosine anneeling
    my_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                                 T_max=500, 
                                                                 eta_min=1e-8, 
                                                                 verbose=True)

# my_lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-6, max_lr=2e-3, 
#                                   step_size_up=2000, step_size_down=None, 
#                                   mode='triangular', gamma=1.0, scale_fn=None, scale_mode='cycle', 
#                                   cycle_momentum=False, base_momentum=0.7, max_momentum=0.9, verbose=True)



if loss_name == "ssim": ssim = True
else: ssim = False
print("Starting training.")



def validate(model, criterion, dataloader, n_val, unet=False, ssim=False):
    model.eval()
    running_val_loss, running_mse_loss, running_ssim_loss, running_l1_loss = 0, 0, 0, 0    
    # Losses
    mse_loss = torch.nn.MSELoss()
    l1_loss = torch.nn.L1Loss()
    ssim_loss = pytorch_ssim.SSIM()    
    # Validation
    count = 0
    count_batch = 0
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):

            sequence, target = data
            sequence = sequence.float().cuda()
            target = target.float().cuda()
            
            outputs = model(sequence)
            if unet: target = target[:, 0]
            else: target = target[:, 0, 0]
            # Training loss
            loss = criterion(outputs[:, 0], target)     
            if ssim: loss = -loss
            
            # Evaluation losses
            mse_ = mse_loss(outputs[:, 0], target).item()
            ssim_ = ssim_loss(outputs[:, 0], target).item()
            l1 = l1_loss(outputs[:, 0], target).item()
            
            running_val_loss += loss.item()
            running_l1_loss += l1_
            running_mse_loss += mse_
            running_ssim_loss += ssim_
#             torch.cuda.empty_cache()            
         
            if count > n_val: break
            else: 
                count_batch += 1
                count += len(target)
               
    # Get the average loss per batch
    running_val_loss /= count_batch
    running_mse_loss /= count_batch
    running_ssim_loss /= count_batch
    running_l1_loss /= count_batch
    
#     print("output: ", outputs.shape, "target", target.shape)
    return running_val_loss, running_mse_loss, running_ssim_loss, running_l1_loss, outputs.detach().cpu().numpy()[:5], target.detach().cpu().numpy()[:5]

def train(sequence, target, model, loss, unet, ssim, optimizer):
    
    # Losses
    mse_loss = torch.nn.MSELoss()
    ssim_loss = pytorch_ssim.SSIM() 
    l1_loss = torch.nn.L1Loss()
    
    # Prediction
    outputs = model(sequence)
    
    if unet: target = target[:, 0]
    else: target = target[:, 0, 0]
    loss_value = loss(outputs[:, 0], target)
    if ssim: loss_value = -loss_value

    # Backpropagation
    loss_value.backward()
    optimizer.step()
    optimizer.zero_grad() 
#     torch.cuda.empty_cache()  

    # print statistics
    loss_ = loss_value.item()
    ssim_ = ssim_loss(outputs[:, 0], target).item()
    mse_ = mse_loss(outputs[:, 0], target).item()
    l1_ = l1_loss(outputs[:, 0], target).item()
    return loss_, ssim_, mse_, l1_


# Writer to save results
writer = SummaryWriter(save_path)
if mode == "infinite":
    print('\nInfinite training')
    
    
    iter_limit = 1e5
    iter_limit = int(6e5 / batch_size)
    
    
    count_no_improvement = 0
    best_val = np.inf
    best_train = np.inf
    model.train()
    val_step = 10
    loss_train, ssim_train, mse_train, l1_train = 0, 0, 0, 0
    a = time.time()
    for iteration, data in enumerate(train_loader, 0):
        if iteration > iter_limit: 
            print("Stopped training at 1e5 iterations.")
            break
        sequence, target = data
        sequence = sequence.float().cuda()
        target = target.float().cuda()
        
        loss_, ssim_, mse_, l1_ = train(sequence, target, model, loss, unet, ssim, optimizer)
        
        loss_train += loss_ / val_step
        ssim_train += ssim_ / val_step
        mse_train += mse_ / val_step
        l1_train += l1_ / val_step
        
        
        # Validation
        if iteration % val_step == 0:
            max_mem = torch.cuda.max_memory_allocated()
            loss_val, mse_val, ssim_val, l1_val, pred, gt = validate(model, loss, val_loader, n_val=n_val, unet=unet, ssim=ssim)
            
            # Decrease learning rate when needed
            if lr_scheduler == "plateau":
                my_lr_scheduler.step(loss_val)
            else:
                my_lr_scheduler.step()
            
            writer.add_scalars("Loss: {}".format(loss_name), {"train":loss_train, "validation":loss_val}, iteration)    
            writer.add_scalars("SSIM", {"train":ssim_train, "validation":ssim_val}, iteration)
            writer.add_scalars("MSE", {"train":mse_train, "validation":mse_val}, iteration)
            writer.add_scalars("L1", {"train":l1_train, "validation":l1_val}, iteration)
            writer.add_scalar("Learning rate", get_lr(optimizer), iteration)
            
            if iteration % 20 == 0:
                idx = int(target.shape[1] / 2)
                for k in range(len(pred)):
                    fig = plt.figure(figsize=(12, 6))
                    plt.subplot(121)
                    plt.title("Prediction")
                    plt.axis('off')
                    plt.imshow(pred[k, 0, idx], cmap="magma")
                    plt.subplot(122)
                    plt.title("Ground-truth")
                    plt.axis('off')
                    plt.imshow(gt[k, idx], cmap="magma")

                    writer.add_figure("Sample {}".format(k), fig, global_step=iteration, close=True)
            writer.flush()

            print("Iteration {} {:.2f} sec:\tLoss train:  {:.2e} \tLoss val:  {:.2e} \tL1 train: {:.2e} \tL1 val: {:.2e} \tSSIM train: {:.2e} \tSSIM val: {:.2e}".format(
                                                                                                                    iteration,
                                                                                                                    time.time() - a, 
                                                                                                                    loss_train, loss_val,
                                                                                                                    l1_train, l1_val,
                                                                                                                    ssim_train, ssim_val))
            # Save models
            if loss_val < best_val: 
                count_no_improvement = 0
                best_val = loss_val
                torch.save({
                    'epoch': iteration,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss}, 
                    save_path+ "/best_val_settings.pt")  
                torch.save(model.state_dict(), save_path + "/best_val_model.pt")
            elif count_no_improvement > 5000:
                print("\nEarly stopping")
                break
            elif iteration > 500: 
                count_no_improvement += 1

            if loss_train < best_train:
                best_train = loss_train
                torch.save({
                    'epoch': iteration,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss}, 
                    save_path+ "/best_train_settings.pt")  
                torch.save(model.state_dict(), save_path + "/best_train_model.pt")
                
            # Reset
            torch.cuda.reset_max_memory_allocated()
            loss_train, ssim_train, mse_train, l1_train = 0, 0, 0, 0
            a = time.time()

else:
    n_epochs = 100
    count_no_improvement = 0
    model.train()
    val_step = 10
    iter_limit = 1e5
    loss_train, ssim_train, mse_train, l1_train = 0, 0, 0, 0
    best_val, best_train = np.inf, np.inf
    a = time.time()
    for epoch in range(n_epochs):
        for iteration, data in enumerate(train_loader, 0):
            if iteration + len(train_loader) * epoch > iter_limit: break
            sequence, target = data
            sequence = sequence.float().cuda()
            target = target.float().cuda()

            loss_, ssim_, mse_, l1_ = train(sequence, target, model, loss, unet, ssim, optimizer)

            loss_train += loss_ / val_step
            ssim_train += ssim_ / val_step
            mse_train += mse_ / val_step
            l1_train += l1_ / val_step

            # Validation
            if iteration % val_step == 0:
                max_mem = torch.cuda.max_memory_allocated()
                loss_val, mse_val, ssim_val, l1_val, pred, gt = validate(model, loss, val_loader, n_val=n_val, unet=unet, ssim=ssim)
                
                # Decrease learning rate when needed
                if lr_scheduler == "plateau":
                    my_lr_scheduler.step(loss_val)
                else:
                    my_lr_scheduler.step()

                writer.add_scalars("Loss: {}".format(loss_name), {"train":loss_train, "validation":loss_val}, iteration + len(train_loader) * epoch)    
                writer.add_scalars("SSIM", {"train":ssim_train, "validation":ssim_val}, iteration + len(train_loader) * epoch)
                writer.add_scalars("MSE", {"train":mse_train, "validation":mse_val}, iteration + len(train_loader) * epoch)
                writer.add_scalars("L1", {"train":l1_train, "validation":l1_val}, iteration + len(train_loader) * epoch)
                writer.add_scalar("Learning rate", get_lr(optimizer), iteration + len(train_loader) * epoch)

                if iteration % 20 == 0:
                    idx = int(patch_size / 2)
                    for k in range(len(pred)):
                        fig = plt.figure(figsize=(12, 6))
                        plt.subplot(121)
                        plt.title("Prediction")
                        plt.axis('off')
                        plt.imshow(pred[k, 0, idx], cmap="magma")
                        plt.subplot(122)
                        plt.title("Ground-truth")
                        plt.axis('off')
                        plt.imshow(gt[k, idx], cmap="magma")

                        writer.add_figure("Sample {}".format(k), fig, global_step=iteration + len(train_loader) * epoch, close=True)
                writer.flush()

                print("Iteration {} {:.2f} sec:\tLoss train:  {:.2e} \tLoss val:  {:.2e} \tL1 train: {:.2e} \tL1 val: {:.2e} \tSSIM train: {:.2e} \tSSIM val: {:.2e}".format(
                                                                                                                        iteration + len(train_loader) * epoch,
                                                                                                                        time.time() - a, 
                                                                                                                        loss_train, loss_val,
                                                                                                                        l1_train, l1_val,
                                                                                                                        ssim_train, ssim_val))
                # Save models
                if loss_val < best_val: 
                    count_no_improvement = 0
                    best_val = loss_val
                    torch.save({
                        'epoch': iteration + len(train_loader) * epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss}, 
                        save_path+ "/best_val_settings.pt")  
                    torch.save(model.state_dict(), save_path + "/best_val_model.pt")
                elif count_no_improvement > 2000:
                    print("\nEarly stopping")
                    break
                else: 
                    count_no_improvement += 1

                if loss_train < best_train:
                    best_train = loss_train
                    torch.save({
                        'epoch': iteration + len(train_loader) * epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss}, 
                        save_path+ "/best_train_settings.pt")  
                    torch.save(model.state_dict(), save_path + "/best_train_model.pt")

                # Reset
                torch.cuda.reset_max_memory_allocated()
                loss_train, ssim_train, mse_train, l1_train = 0, 0, 0, 0
                a = time.time()
    
