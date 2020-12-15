"""Shape induced MCD
"""

import random
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import itertools
import tqdm
from data.dataset import UnpairedDataset
from models.unet import UNet
from utils.losses import DiceLoss, WassersteinLoss
from utils.trainer import train_segmentation_net, train_mcd


# Training variables
seed = 20
torch.manual_seed(seed)
np.random.seed(seed)

# 2 for training on sobel edges, 0 for raw image
IMAGE_INDEX = 2
SLICED = True
project_name = 'unet_sobel_swd_eadan_in_2_no_reduce'
result_path = './results'
## init nets
net = UNet(1, 1)


# use this for all experiments from here on out!
batch_size = 20
num_train = 5000
num_val = 47
num_test_gold = 250
learning_rate = 0.0002
epochs_mcd = 20
is_save = True

## loss functions
criterion_wstein = WassersteinLoss(sliced=SLICED)

def criterion_cls(pred, label):
    return nn.BCELoss()(pred, label) + DiceLoss()(pred, label)


## datasets and loaders
train_scan_dataset = UnpairedDataset('../', 'ct_sag_kr/train', 'mr_sag_kr/train')
# val_scan_dataset = UnpairedDataset('../', 'ct_sag_kr/test', 'mr_sag_kr/test')
test_scan_dataset = UnpairedDataset('../', 'ct_sag_kr/test', 'mr_sag_kr/test_gold')

scan_dataset, _ = random_split(train_scan_dataset, [num_train, len(train_scan_dataset) - num_train])
scan_dataset_val, scan_dataset_test = random_split(test_scan_dataset, [num_val, len(test_scan_dataset) - num_val])

# scan_dataset_test, _ = random_split(test_scan_dataset, [num_test_gold, len(test_scan_dataset) - num_test_gold])

train_loader = DataLoader(scan_dataset, batch_size=batch_size, num_workers=5, pin_memory=True)
val_loader = DataLoader(scan_dataset_val, batch_size=1, shuffle=False, num_workers=5, pin_memory=True)
test_loader = DataLoader(scan_dataset_test, batch_size=1, shuffle=False, num_workers=5, pin_memory=True)

net.cuda()

# weight_decay=0.0005

## init optimisers
optimiser_enc = optim.Adam(itertools.chain(net.inc.parameters(), 
                                           net.down1.parameters(), 
                                           net.down2.parameters(), 
                                           net.down3.parameters(), 
                                           net.down4.parameters()), lr=learning_rate)

optimiser_dec = optim.Adam(itertools.chain(net.up1.parameters(), 
                                           net.up2.parameters(), 
                                           net.up3.parameters(), 
                                           net.up4.parameters(), 
                                           net.outc.parameters(), 
                                           net.final.parameters()), lr=learning_rate)

print('-- Project name ', project_name)

# MCD loss
net_train_dice_a = []
net_val_loss_a = []
net_val_dice_a = []
net_val_loss_b = []
net_val_dice_b = []

net_test_loss_a = None
net_test_dice_a = None
net_test_loss_b = None
net_test_dice_b = None


print('MCD Training...')
# Part 1 - Shape induced MCD
for i in range(epochs_mcd):
    train_dice_a = train_mcd(net, train_loader, IMAGE_INDEX, 1, 
                             criterion_cls, criterion_wstein, optimiser_enc, optimiser_dec, epoch=i)
    
    dice_a, loss_a = train_segmentation_net(net, val_loader, 'A', IMAGE_INDEX, 1, 
                                            criterion_cls, is_train=False, name='MCD val A', epoch=i)
    
    dice_b, loss_b = train_segmentation_net(net, val_loader, 'B', IMAGE_INDEX, 1, 
                                            criterion_cls, is_train=False, name='MCD val B', epoch=i)
    
    net_train_dice_a.append(train_dice_a)
    net_val_loss_a.append(loss_a)
    net_val_dice_a.append(dice_a)
    net_val_loss_b.append(loss_b)
    net_val_dice_b.append(dice_b)

dice_a, loss_a = train_segmentation_net(net, test_loader, 'A', IMAGE_INDEX, 1, 
                                        criterion_cls, is_train=False, name='MCD test A ' + project_name)
    
dice_b, loss_b = train_segmentation_net(net, test_loader, 'B', IMAGE_INDEX, 1, 
                                        criterion_cls, is_train=False, name='MCD test B ' + project_name)

net_test_loss_a = loss_a
net_test_dice_a = dice_a
net_test_loss_b = loss_b
net_test_dice_b = dice_b


# save results here
if is_save == True:
    save_path = os.path.join(result_path, project_name)

    if os.path.isdir(save_path) is False:
        os.makedirs(save_path)

    np.savez(os.path.join(save_path, 'params'), 
             num_train=num_train,
             num_val=num_val,
             num_test=num_test_gold,
             epochs_mcd=epochs_mcd,
             learning_rate=learning_rate,
             batch_size=batch_size,

             net_train_dice_a=net_train_dice_a,
             net_val_loss_a=net_val_loss_a,
             net_val_dice_a=net_val_dice_a,
             net_val_loss_b=net_val_loss_b,
             net_val_dice_b=net_val_dice_b, 

             net_test_loss_a = net_test_loss_a,
             net_test_dice_a = net_test_dice_a,
             net_test_loss_b = net_test_loss_b,
             net_test_dice_b = net_test_dice_b)

    torch.save(net.state_dict(), os.path.join(save_path, 'net'))
