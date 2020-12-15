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
from utils.losses import DiceLoss
from utils.lovasz_losses import symmetric_lovasz

# metrics here
from utils.metrics import compute_dice_metric

# init seed
seed = 20
torch.manual_seed(seed)
np.random.seed(seed)

project_name = 'dsbn_final'
project_description = 'Train domain specific batch norm'
result_path = './results'

batch_size = 32
num_a_train = 7770
num_a_val = 250
num_b_train = 5300
num_b_val = 47
learning_rate = 0.0002
epochs = 60

# A - CT, B - MR
DOMAIN_A = 0
DOMAIN_B = 1


def criterion_seg(pred, label):
    return nn.BCELoss()(pred, label) + DiceLoss()(pred, label)


# CT loader
train_a_dataset = UnpairedDataset('../', path_a='ct_sag_kr/train', path_b=None)
test_a_dataset = UnpairedDataset('../', path_a='ct_sag_kr/test', path_b=None)
num_a_test = len(test_a_dataset) - num_a_val

train_a_dataset, _ = random_split(train_a_dataset, [num_a_train, len(train_a_dataset) - num_a_train])
val_a_dataset, test_a_dataset = random_split(test_a_dataset, [num_a_val, num_a_test])

train_a_loader = DataLoader(train_a_dataset, batch_size=batch_size, num_workers=5, pin_memory=True)
val_a_loader = DataLoader(val_a_dataset, batch_size=1, shuffle=False, num_workers=5, pin_memory=True)
test_a_loader = DataLoader(test_a_dataset, batch_size=1, shuffle=False, num_workers=5, pin_memory=True)


# MR loader
train_b_dataset = UnpairedDataset('../', path_a='mr_sag_kr/train_reduced', path_b=None)
test_b_dataset = UnpairedDataset('../', path_a='mr_sag_kr/test_gold', path_b=None)
num_b_test = len(test_b_dataset) - num_b_val

train_b_dataset, _ = random_split(train_b_dataset, [num_b_train, len(train_b_dataset) - num_b_train])
val_b_dataset, test_b_dataset = random_split(test_b_dataset, [num_b_val, num_b_test])

train_b_loader = DataLoader(train_b_dataset, batch_size=batch_size, num_workers=5, pin_memory=True)
val_b_loader = DataLoader(val_b_dataset, batch_size=1, shuffle=False, num_workers=5, pin_memory=True)
test_b_loader = DataLoader(test_b_dataset, batch_size=1, shuffle=False, num_workers=5, pin_memory=True)


# net and optimizer
ds_unet = UNet(1, 1, domain_specific=True)
ds_unet.cuda()
labeller = UNet(1, 1)
# import weights here...
labeller_path = './results/unet_sobel_eadan_in/net'
labeller.load_state_dict(torch.load(os.path.join(labeller_path), 
                               map_location=lambda storage, loc: storage))
labeller.cuda()

optimiser = optim.Adam(ds_unet.parameters(), lr=learning_rate)

print('Project name ', project_name)

train_dices = []
train_losses = []
val_a_dices = []
val_a_losses = []

val_b_dices = []
val_b_losses = []


for i in range(epochs):
    ds_unet.set_domain(DOMAIN_A)
    train_dice, train_loss = train_segmentation_net(ds_unet, train_a_loader, 'A', 0, 1, 
                                                    criterion_seg, opt=optimiser, 
                                                    is_train=True, name='dsbn a train', epoch=i)
    
    val_dice, val_loss = train_segmentation_net(ds_unet, val_a_loader, 'A', 0, 1, 
                                                criterion_seg, is_train=False, name='dsbn a val', epoch=i)
    
    train_dices.append(train_dice)
    train_losses.append(train_loss)
    val_a_dices.append(val_dice)
    val_a_losses.append(val_loss)
    
    ds_unet.set_domain(DOMAIN_B)
    
    train_segmentation_net(ds_unet, train_b_loader, 'A', 0, 1, criterion_seg, opt=optimiser, 
                           is_train=True, labeller=labeller, threshold=0.95, name='dsbn b train', epoch=i, verbose=False)
    
    val_dice, val_loss = train_segmentation_net(ds_unet, val_b_loader, 'A', 0, 1, 
                                                criterion_seg, is_train=False, name='dsbn b val', epoch=i)
    
    val_b_dices.append(val_dice)
    val_b_losses.append(val_loss)
    

ds_unet.set_domain(DOMAIN_A)
test_a_dice, test_a_loss = train_segmentation_net(ds_unet, test_a_loader, 'A', 0, 1, criterion_seg, 
                                                  is_train=False, name='dsbn a test', epoch=0)

ds_unet.set_domain(DOMAIN_B)
test_b_dice, test_b_loss = train_segmentation_net(ds_unet, test_b_loader, 'A', 0, 1, criterion_seg, 
                                                  is_train=False, name='dsbn b test', epoch=0)

save_path = os.path.join(result_path, project_name)

if os.path.isdir(save_path) is False:
    os.makedirs(save_path)
    
all_results = {
    "project_name": project_name,
    "project_description": project_description,
    "epochs": epochs,
    "num_a_train": num_a_train,
    "num_a_val": num_a_val,
    "num_a_test": num_a_test,
    "num_b_train": num_b_train,
    "num_b_val": num_b_val,
    "num_b_test": num_b_test,
    "learning_rate": learning_rate,
    "batch_size": batch_size,
    "train_dice": train_dices,
    "train_loss": train_losses,
    "val_a_dice": val_a_dices,
    "val_a_loss": val_a_losses,
    "val_b_dice": val_b_dices,
    "val_b_loss": val_b_losses,
    "test_a_dice": test_a_dice,
    "test_a_loss": test_a_loss,
    "test_b_dice": test_b_dice,
    "test_b_loss": test_b_loss,
}

np.savez(os.path.join(save_path, 'params'), **all_results)
torch.save(ds_unet.state_dict(), os.path.join(save_path, 'ds_unet'))

