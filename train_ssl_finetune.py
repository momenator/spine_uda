import glob
import random
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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
from models.unet import UNet, UNetDense

# loss functions here
from utils.losses import DiceLoss, WassersteinLoss, entropy_loss

# metrics here
from utils.metrics import compute_dice_metric

# scheduler
# from utils.schedulers import LambdaLR

# init seed
seed = 20
torch.manual_seed(seed)
np.random.seed(seed)

project_name = 'unet_dense_ssl_pretraining'
result_path = './results'

batch_a_size = 10
batch_b_size = 10
batch_c_size = 10

num_a_train = 7000
num_a_val = 400
num_a_test = 400

num_b_train = 9000
num_b_val = 250
num_b_test = 250

is_finetuning = False
num_c_train = 100
num_c_val = 75
num_c_test = 100

learning_rate = 0.0002
epochs_pseudo = 5
epochs_ssl = 50
epochs_finetune = 2


def criterion(pred, label):
    return nn.BCELoss()(pred, label) + DiceLoss()(pred, label)


def train_net(net, loader, data_key, image_idx, target_idx, epoch, is_train=True, opt=None, labeller=None, name='train'):
    dice_scores = []
    losses = []
    count = 0
    
    for i, data in enumerate(tqdm.tqdm(loader)):                
        image = data[data_key][image_idx].cuda()
        
        if labeller == None:
            target = data[data_key][target_idx].cuda()
        else:
            labeller.eval()
            with torch.no_grad():
                target = labeller.upsample(*(labeller.downsample(image)))
                target = torch.round(target).detach().cuda()
                
        if is_train:
            net.train()
            opt.zero_grad()
            # format for prediction is fixed
            pred = net.upsample(*(net.downsample(image)))
            loss = criterion(pred, target)
            pred = torch.round(pred)
            loss.backward()
            opt.step()
            loss = loss.item()
        else:
            net.eval()
            # do validation on test set here!
            with torch.no_grad():
                pred = net.upsample(*(net.downsample(image)))
                loss = criterion(pred, target).item()
                pred = torch.round(pred)
        
        dice_score = compute_dice_metric(pred, target).item()
        dice_scores.append(dice_score)
        losses.append(loss)
                        
        count += 1
    
    # print dice scores here!
    mean_dice = np.mean(dice_scores)
    mean_loss = np.mean(losses)
    
    print('{} - epoch {} - Avg dice: {}, Avg loss: {}'.format(name, epoch, mean_dice, mean_loss))
            
    # return none for the time being
    return mean_dice, mean_loss


# 3 pairs of dataset, CT (Train / test), MRI (Train/test), Siegen public MRI (Train/test)
train_a_dataset = UnpairedDataset('../', path_a='ct_sag_kr/train', path_b=None)
test_a_dataset = UnpairedDataset('../', path_a='ct_sag_kr/test', path_b=None)

train_b_dataset = UnpairedDataset('../', path_a='mr_sag_kr/train', path_b=None)
test_b_dataset = UnpairedDataset('../', path_a='mr_sag_kr/test', path_b=None)

train_c_dataset = UnpairedDataset('../', path_a='siegen/train', path_b=None)
test_c_dataset = UnpairedDataset('../', path_a='siegen/test', path_b=None)


train_a_dataset, _ = random_split(train_a_dataset, [num_a_train, len(train_a_dataset) - num_a_train])
val_a_dataset, test_a_dataset, _ = random_split(test_a_dataset, 
                                                [num_a_val, num_a_test, len(test_a_dataset) - (num_a_val + num_a_test)])

train_b_dataset, _ = random_split(train_b_dataset, [num_b_train, len(train_b_dataset) - num_b_train])
val_b_dataset, test_b_dataset, _ = random_split(test_b_dataset, 
                                                [num_b_val, num_b_test, len(test_b_dataset) - (num_b_val + num_b_test)])

train_c_dataset, _ = random_split(train_c_dataset, [num_c_train, len(train_c_dataset) - num_c_train])
val_c_dataset, test_c_dataset, _ = random_split(test_c_dataset, 
                                                [num_c_val, num_c_test, len(test_c_dataset) - (num_c_val + num_c_test)])


train_a_loader = DataLoader(train_a_dataset, batch_size=batch_a_size, num_workers=5, pin_memory=True)
val_a_loader = DataLoader(val_a_dataset, batch_size=1, shuffle=False, num_workers=5, pin_memory=True)
test_a_loader = DataLoader(test_a_dataset, batch_size=1, shuffle=False, num_workers=5, pin_memory=True)

train_b_loader = DataLoader(train_b_dataset, batch_size=batch_b_size, num_workers=5, pin_memory=True)
val_b_loader = DataLoader(val_b_dataset, batch_size=1, shuffle=False, num_workers=5, pin_memory=True)
test_b_loader = DataLoader(test_b_dataset, batch_size=1, shuffle=False, num_workers=5, pin_memory=True)

train_c_loader = DataLoader(train_c_dataset, batch_size=batch_c_size, num_workers=5, pin_memory=True)
val_c_loader = DataLoader(val_c_dataset, batch_size=1, shuffle=False, num_workers=5, pin_memory=True)
test_c_loader = DataLoader(test_c_dataset, batch_size=1, shuffle=False, num_workers=5, pin_memory=True)

pseudo = UNetDense(1, 1)
net = UNetDense(1, 1, norm_layer=nn.BatchNorm2d, track_running_stats=False, affine=True)

pseudo.cuda()
net.cuda()

# list optimisers here...
# single optimiser variant 1

optimiser_ps = optim.Adam(pseudo.parameters(), lr=learning_rate)
optimiser_net = optim.Adam(net.parameters(), lr=learning_rate)

print('Project name ', project_name)
print('Learning rate ', learning_rate)

pseudo_train_a_dice = []
pseudo_train_a_loss = []
pseudo_val_a_dice = []
pseudo_val_a_loss = []
pseudo_val_b_dice = []
pseudo_val_b_loss = []

parent_train_b_dice = []
parent_train_b_loss = []
parent_val_b_dice = []
parent_val_b_loss = []

parent_train_finetune_dice = []
parent_train_finetune_loss = []
parent_val_finetune_dice = []
parent_val_finetune_loss = []
parent_val_b_finetune_dice = []
parent_val_b_finetune_loss = []


for i in range(epochs_pseudo):
    # train pseudolabeller here
    t_dice, t_loss = train_net(pseudo, train_a_loader, 'A', 2, 1, i, is_train=True, opt=optimiser_ps, name='Pseudo train a')
    v_dice, v_loss = train_net(pseudo, val_a_loader, 'A', 2, 1, i, is_train=False, name='Pseudo val a')
    v_b_train, v_b_loss = train_net(pseudo, val_b_loader, 'A', 2, 1, i, is_train=False, name='Pseudo val b')
    
    pseudo_train_a_dice.append(t_dice)
    pseudo_train_a_loss.append(t_loss)
    pseudo_val_a_dice.append(v_dice)
    pseudo_val_a_loss.append(v_loss)
    pseudo_val_b_dice.append(v_b_train)
    pseudo_val_b_loss.append(v_b_loss)

    
for i in range(epochs_ssl):
    # train main network here
    t_dice, t_loss = train_net(net, train_b_loader, 'A', 0, 1, i, is_train=True, 
                               opt=optimiser_net, labeller=pseudo, name='Parent train b')
    v_dice, v_loss = train_net(net, val_b_loader, 'A', 0, 1, i, is_train=False, 
                               labeller=pseudo, name='Parent val b')
    
    parent_train_b_dice.append(t_dice)
    parent_train_b_loss.append(t_loss)
    parent_val_b_dice.append(v_dice)
    parent_val_b_loss.append(v_loss)

# test pre_tuning
dice_pretuning, loss_pretuning = train_net(net, test_b_loader, 'A', 0, 1, 0, is_train=False, name='pretuning test b')

if is_finetuning:
    for i in range(epochs_finetune):
        # finetune main network here
        t_dice, t_loss = train_net(net, train_c_loader, 'A', 0, 1, i, is_train=True, 
                                   opt=optimiser_net, name='Finetune train')
        v_dice, v_loss = train_net(net, val_c_loader, 'A', 0, 1, i, is_train=False, name='Finetune val')
        v_b_dice, v_b_loss = train_net(net, val_b_loader, 'A', 0, 1, i, is_train=False, name='Parent val b')

        parent_train_finetune_dice.append(t_dice)
        parent_train_finetune_loss.append(t_loss)
        parent_val_finetune_dice.append(v_dice)
        parent_val_finetune_loss.append(v_loss)
        parent_val_b_finetune_dice.append(v_b_dice)
        parent_val_b_finetune_loss.append(v_b_loss)

    # final test - post tuning - does it help?
    dice_posttuning, loss_posttuning = train_net(net, test_b_loader, 'A', 0, 1, 0, is_train=False, name='postuning test b')


# save results here
save_path = os.path.join(result_path, project_name)

if os.path.isdir(save_path) is False:
    os.makedirs(save_path)

all_results = {
    "epochs_pseudo": epochs_pseudo,
    "epochs_ssl": epochs_ssl,
    "epochs_finetune": epochs_finetune,
    "learning_rate": learning_rate,
    "batch_size": batch_a_size,

    "pseudo_train_a_dice": pseudo_train_a_dice,
    "pseudo_train_a_loss": pseudo_train_a_loss,
    "pseudo_val_a_dice": pseudo_val_a_dice,
    "pseudo_val_a_loss": pseudo_val_a_loss,
    "pseudo_val_b_dice": pseudo_val_b_dice,
    "pseudo_val_b_loss": pseudo_val_b_loss,

    "parent_train_b_dice": parent_train_b_dice,
    "parent_train_b_loss": parent_train_b_loss,
    "parent_val_b_dice": parent_val_b_dice,
    "parent_val_b_loss": parent_val_b_loss,
    "dice_pretuning": dice_pretuning, 
    "loss_pretuning": loss_pretuning
}


if is_finetuning:
    all_results['parent_train_finetune_dice'] = parent_train_finetune_dice
    all_results['parent_train_finetune_loss'] = parent_train_finetune_loss
    all_results['parent_val_finetune_dice'] = parent_val_finetune_dice
    all_results['parent_val_finetune_loss'] = parent_val_finetune_loss
    all_results['parent_val_b_finetune_dice'] = parent_val_b_finetune_dice
    all_results['parent_val_b_finetune_loss'] = parent_val_b_finetune_loss
    all_results['dice_posttuning'] = dice_posttuning
    all_results['loss_posttuning'] = loss_posttuning

np.savez(os.path.join(save_path, 'params'), **all_results)
torch.save(net.state_dict(), os.path.join(save_path, 'net'))
