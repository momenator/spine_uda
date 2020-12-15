import glob
import random
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from data.dataset import UnpairedDataset
from models.unet import UNet
from utils.trainer import train_segmentation_net

# loss functions here
from utils.losses import DiceLoss, WassersteinLoss, entropy_loss
from utils.lovasz_losses import symmetric_lovasz

# metrics here
from utils.metrics import compute_dice_metric

# init seed
seed = 32
torch.manual_seed(seed)
np.random.seed(seed)

project_name = 'unet_shape_model_2'
project_description = 'UNet (original) for learning vertebral shape model with (4,4) corrupt kernel and mse loss'
result_path = './results'

batch_size = 32
num_train = 6500
num_val = 250
learning_rate = 0.001
epochs = 8

def criterion_seg(pred, label):
    return nn.MSELoss()(pred, label)


train_dataset = UnpairedDataset('../', path_a='ct_sag_kr/train', path_b=None, corrupt=True, augment=True)
test_dataset = UnpairedDataset('../', path_a='ct_sag_kr/test', path_b=None, corrupt=True)
num_test = len(test_dataset) - num_val

train_dataset, _ = random_split(train_dataset, [num_train, len(train_dataset) - num_train])
val_dataset, test_dataset = random_split(test_dataset, [num_val, num_test])

train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=5, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=5, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=5, pin_memory=True)

refine_net = UNet(1, 1, norm_layer=nn.BatchNorm2d, affine=True, track_running_stats=True)
refine_net.cuda()

# list optimisers here...
# single optimiser variant 1

optimiser_ref = optim.Adam(refine_net.parameters(), lr=learning_rate)

print('Project name ', project_name)

train_dices = []
train_losses = []
val_dices = []
val_losses = []

for i in range(epochs):
    train_dice, train_loss = train_segmentation_net(refine_net, train_loader, 'A', 3, 1, 
                                                    criterion_seg, opt=optimiser_ref, is_train=True, name='Refine train', epoch=i)
    
    val_dice, val_loss = train_segmentation_net(refine_net, val_loader, 'A', 3, 1, 
                                                criterion_seg, is_train=False, name='Refine val', epoch=i)
    
    train_dices.append(train_dice)
    train_losses.append(train_loss)
    val_dices.append(val_dice)
    val_losses.append(val_loss)
    
    
test_dice, test_loss = train_segmentation_net(refine_net, test_loader, 'A', 3, 1, 
                                              criterion_seg, is_train=False, name='Refine test', epoch=0)

save_path = os.path.join(result_path, project_name)

if os.path.isdir(save_path) is False:
    os.makedirs(save_path)
    
all_results = {
    "project_name": project_name,
    "project_description": project_description,
    "epochs": epochs,
    "num_train": num_train,
    "num_val": num_val,
    "num_test": num_test,
    "learning_rate": learning_rate,
    "batch_size": batch_size,
    "train_dice": train_dices,
    "train_loss": train_losses,
    "val_dice": val_dices,
    "val_loss": val_losses,
    "test_dice": test_dice,
    "test_loss": test_loss,
}

np.savez(os.path.join(save_path, 'params'), **all_results)
torch.save(refine_net.state_dict(), os.path.join(save_path, 'refiner'))

