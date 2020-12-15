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
from models.unet import UNet, UNetWavelet, UNetDense

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

project_name = 'exp17_unet_dsbn2_ssl'
result_path = './results'
batch_size = 14

num_train = 7000
num_val = 250
num_test = 250

learning_rate = 0.0002
iteration = 0
epochs_init = 50
epochs_decay = 0
epochs_pseudo = 5
epochs = epochs_init + epochs_decay + epochs_pseudo

DOMAIN_A = 0
DOMAIN_B = 1


discrepancy_loss = WassersteinLoss()
# loss_fn = nn.BCEWithLogitsLoss()
loss_fn = nn.L1Loss()


def criterion(pred, label):
    return nn.BCELoss()(pred, label) + DiceLoss()(pred, label)


# should we use 
def validate_net(net, loader, name='Test', epoch=None, is_save=True, limit=None, display=False):
    net.eval()
    dices_a = []
    dices_b = []
    losses_a = []
    losses_b = []
    
    count = 0
    
    for j, data in enumerate(loader):
        
        if limit != None and count >= limit:
            break
            
        image_a = data['A'][0].cuda()
        target_a = data['A'][1].cuda()
        
        image_b = data['B'][0].cuda()
        target_b = data['B'][1].cuda()
        
        # do validation on test set here!
        with torch.no_grad():
            if torch.max(target_a) != 0:
                # set domain here...
                net.set_domain(DOMAIN_A)
                res_a = net.downsample(image_a)      
                pred = net.upsample(*res_a)
#                 pred = net(image_a)
                loss = criterion(pred, target_a).item()
                pred = torch.round(pred)
                dice_score = compute_dice_metric(pred, target_a).item()
                dices_a.append(dice_score)
                losses_a.append(loss)
            
            if torch.max(target_b) != 0:
                net.set_domain(DOMAIN_B)
                res_b = net.downsample(image_b)      
                pred = net.upsample(*res_b)
#                 pred = net(image_b)
                loss = criterion(pred, target_b).item()
                pred = torch.round(pred)
                dice_score = compute_dice_metric(pred, target_b).item()
                dices_b.append(dice_score)
                losses_b.append(loss)
                        
        count += 1
    
    # print dice scores here!
    mean_dice_a = np.mean(dices_a)
    mean_dice_b = np.mean(dices_b)
    
    mean_loss_a = np.mean(losses_a)
    mean_loss_b = np.mean(losses_b)
    
    print('{} - Avg dice A: {}, Avg dice B: {}, Avg loss A: {}, Avg loss B: {}'.format(name, 
                                                                                       mean_dice_a, 
                                                                                       mean_dice_b, 
                                                                                       mean_loss_a, 
                                                                                       mean_loss_b))
            
    # return none for the time being
    return mean_dice_a, mean_dice_b, mean_loss_a, mean_loss_b


# loader from siegen dataset for validation
train_scan_dataset = UnpairedDataset('../', 'ct_sag_kr/train', 'mr_sag_kr/train')
test_scan_dataset = UnpairedDataset('../', 'ct_sag_kr/test', 'mr_sag_kr/test')

scan_dataset, _ = random_split(train_scan_dataset, [num_train, len(train_scan_dataset) - num_train])
scan_dataset_test, scan_dataset_val, _ = random_split(test_scan_dataset, 
                                                      [num_val, num_test, len(test_scan_dataset) - (num_val + num_test)])

train_loader = DataLoader(scan_dataset, batch_size=batch_size, num_workers=5, pin_memory=True)
val_loader = DataLoader(scan_dataset_val, batch_size=1, shuffle=False, num_workers=5, pin_memory=True)
test_loader = DataLoader(scan_dataset_test, batch_size=1, shuffle=False, num_workers=5, pin_memory=True)


pseudo = UNet(1, 1)
net = UNet(1, 1, domain_specific=True)

pseudo.cuda()
net.cuda()

# list optimisers here...
# single optimiser variant 1

optimiser_ps = optim.Adam(pseudo.parameters(), lr=learning_rate)
optimiser_net = optim.Adam(net.parameters(), lr=learning_rate)

print('Project name ', project_name)
print('Learning rate ', learning_rate)
print('Epochs ', epochs)

train_loss = []
train_loss_rec = []
train_loss_seg = []

train_loss_seg_a = []
train_loss_seg_b = []

train_dice = []
val_loss_a = []
val_dice_a = []
val_loss_b = []
val_dice_b = []

for e in range(epochs):
    epoch_train_loss_rec = []
    epoch_train_loss_seg = []
    
    dice_scores = []
    net.train()
    pseudo.train()
    
    print('Epoch ', e)
    
    for i, data in enumerate(tqdm.tqdm(train_loader)):
        iteration += batch_size
        
        optimiser_ps.zero_grad()
        optimiser_net.zero_grad()
                
        # either train pseudolabeller or the net
        # first 10 epochs train the pseudo labeller on edges
        if e < epochs_pseudo:
            edges_a = data['A'][2].cuda()
            target_a = data['A'][1].cuda()
            
            res_pseudo = pseudo.downsample(edges_a)        
            pred_seg_a = pseudo.upsample(*res_pseudo)
#             pred_seg_a = pseudo(edges_a)
            loss_seg_a = criterion(pred_seg_a, target_a)

            loss_seg_a.backward()        
            optimiser_ps.step()
        
        else:
            pseudo.eval()
            image_a = data['A'][0].cuda()
            target_a = data['A'][1].cuda()
            
            image_b = data['B'][0].cuda()
            edges_b = data['B'][2].cuda()
            pseudo_b = pseudo.downsample(edges_b)
            pred_pseudo_b = pseudo.upsample(*pseudo_b)
#             pred_pseudo_b = pseudo(edges_b)
            target_b = torch.round(pred_pseudo_b).detach().cuda()
            
            net.set_domain(DOMAIN_A)
            res_a = net.downsample(image_a)        
            pred_seg_a = net.upsample(*res_a)
            
            net.set_domain(DOMAIN_B)
            res_b = net.downsample(image_b)        
            pred_seg_b = net.upsample(*res_b)

            loss_seg_a = criterion(pred_seg_a, target_a)
            loss_seg_b = criterion(pred_seg_b, target_b)

            loss = loss_seg_a + loss_seg_b
            
            loss.backward()        
            optimiser_net.step()
            
        # dice_score = dice_coeff(torch.round(pred), l).item()
        # epoch_train_loss_rec.append(loss_recon.item())
        epoch_train_loss_seg.append(loss_seg_a.item())
        
    # mean_loss_rec = np.mean(epoch_train_loss_rec)
    mean_loss_seg = np.mean(epoch_train_loss_seg)
    
    # print('Train A - avg seg:{}'.format(np.mean(epoch_train_loss_seg)))
    
    print('Train A - avg seg: {}'.format(mean_loss_seg))

    if e >= epochs_pseudo:
        dice_a, dice_b, loss_a, loss_b = validate_net(net=net, loader=val_loader, name='Validation ', epoch=str(e))
        val_loss_a.append(loss_a)
        val_dice_a.append(dice_a)
        val_loss_b.append(loss_b)
        val_dice_b.append(dice_b)
    
    # update learning rate
    # scheduler.step()
    
dice_a, dice_b, loss_a, loss_b = validate_net(net=net, loader=test_loader, name='Test ', epoch='final')


# save results here
save_path = os.path.join(result_path, project_name)

if os.path.isdir(save_path) is False:
    os.makedirs(save_path)

np.savez(os.path.join(save_path, 'params'), 
         num_train=num_train,
         num_val=num_val,
         num_test=num_test,
         epochs=epochs,
         learning_rate=learning_rate,
         batch_size=batch_size,
         val_dice_a=val_dice_a, 
         val_dice_b=val_dice_b,
         val_loss_a=val_loss_a, 
         val_loss_b=val_loss_b,
         test_dice_a=dice_a, 
         test_dice_b=dice_b,
         test_loss_a=loss_a, 
         test_loss_b=loss_b)

torch.save(net.state_dict(), os.path.join(save_path, 'net'))
