import glob
import random
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
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
from models.unet import UNet, UNetDense, UNetWavelet, AutoEncoder

# loss functions here
from utils.losses import DiceLoss, WassersteinLoss, entropy_loss
from utils.lovasz_losses import lovasz_softmax

# metrics here
from utils.metrics import compute_dice_metric

# scheduler
from torch.optim.lr_scheduler import StepLR


# init seed
seed = 19
torch.manual_seed(seed)
np.random.seed(seed)

project_name = 'shape_induced_dsbn_wasserstein_exp2'
project_description = 'testing shape-induced network + dsbn + wasserstein loss + semi-supervised'
result_path = './results'

batch_a_size = 25
batch_b_size = 25

num_a_train = 1000
num_a_val = 400
num_a_test = 400

num_a_b_train = 7000
num_b_val = 250
num_b_test = 250
num_b_test_gold = 200

learning_rate = 0.0002
epochs_pseudo = 10
epochs_ssl = 10

DOMAIN_A = 0
DOMAIN_B = 1


def criterion_seg(pred, label):
    return nn.BCELoss()(pred, label) + DiceLoss()(pred, label)


def criterion_reg(pred, label):
    return nn.MSELoss()(pred, label)


def train_net(net, 
              loader, 
              data_key, 
              image_idx, 
              target_idx, 
              epoch, 
              criterion, 
              is_train=True, 
              opt=None, 
              labeller=None,
              refiner=None,
              scheduler=None,
              name='train'):
    
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
                target = labeller(image).detach().cuda()
            
            if refiner != None:
                refiner.eval()
                with torch.no_grad():
                    target_ref = refiner(target).cuda()
                    target = torch.round((target_ref + target) / 2)
            else:
                target = torch.round(target)
                
        if is_train:
            net.train()
            opt.zero_grad()
            # format for prediction is fixed
            # pred = net.upsample(*(net.downsample(image)))
            pred = net(image)
            loss = criterion(pred, target)
            pred = torch.round(pred)
            loss.backward()
            opt.step()
            loss = loss.item()
        else:
            net.eval()
            # do validation on test set here!
            with torch.no_grad():
                # pred = net.upsample(*(net.downsample(image)))
                pred = net(image)
                
                # if refiner is present, use it to refine current prediction
                if refiner != None:
                    refiner.eval()
                    with torch.no_grad():
                        pred_ref = refiner(torch.round(pred))
                        pred = (pred + pred_ref)/2

                pred = torch.round(pred).cuda()
                loss = criterion(pred, target).item()
                pred = torch.round(pred)
        
        dice_score = compute_dice_metric(pred, target).item()
        dice_scores.append(dice_score)
        losses.append(loss)
                        
        count += 1
    
    if scheduler != None:
        print(scheduler.get_last_lr())
        scheduler.step()
    
    # print dice scores here!
    mean_dice = np.mean(dice_scores)
    mean_loss = np.mean(losses)
    
    print('{} - epoch {} - Avg dice: {}, Avg loss: {}'.format(name, epoch, mean_dice, mean_loss))
            
    # return none for the time being
    return mean_dice, mean_loss


# 3 pairs of dataset, CT (Train / test), MRI (Train/test), Siegen public MRI (Train/test)
train_a_dataset = UnpairedDataset('../', path_a='ct_sag_kr/train', path_b=None, corrupt=True)
test_a_dataset = UnpairedDataset('../', path_a='ct_sag_kr/test', path_b=None, corrupt=True)

train_a_b_dataset = UnpairedDataset('../', path_a='ct_sag_kr/train', path_b='mr_sag_kr/train')

test_b_dataset = UnpairedDataset('../', path_a='mr_sag_kr/test', path_b=None)
test_b_dataset_gold = UnpairedDataset('../', path_a='mr_sag_kr/test_gold', path_b=None)

# train_c_dataset = UnpairedDataset('../', path_a='siegen/train', path_b=None)
# test_c_dataset = UnpairedDataset('../', path_a='siegen/test', path_b=None)


train_a_dataset, _ = random_split(train_a_dataset, [num_a_train, len(train_a_dataset) - num_a_train])
val_a_dataset, test_a_dataset, _ = random_split(test_a_dataset, 
                                                [num_a_val, num_a_test, len(test_a_dataset) - (num_a_val + num_a_test)])

train_a_b_dataset, _ = random_split(train_a_b_dataset, [num_a_b_train, len(train_a_b_dataset) - num_a_b_train])

val_b_dataset, test_b_dataset, _ = random_split(test_b_dataset, 
                                                [num_b_val, num_b_test, len(test_b_dataset) - (num_b_val + num_b_test)])

test_b_dataset_gold, _ = random_split(test_b_dataset_gold, [num_b_test_gold, len(test_b_dataset_gold) - num_b_test_gold])

train_a_loader = DataLoader(train_a_dataset, batch_size=batch_a_size, num_workers=5, pin_memory=True)
val_a_loader = DataLoader(val_a_dataset, batch_size=1, shuffle=False, num_workers=5, pin_memory=True)
test_a_loader = DataLoader(test_a_dataset, batch_size=1, shuffle=False, num_workers=5, pin_memory=True)

train_a_b_loader = DataLoader(train_a_b_dataset, batch_size=batch_b_size, num_workers=5, pin_memory=True)
val_b_loader = DataLoader(val_b_dataset, batch_size=1, shuffle=False, num_workers=5, pin_memory=True)
test_b_loader = DataLoader(test_b_dataset, batch_size=1, shuffle=False, num_workers=5, pin_memory=True)
test_b_loader_gold = DataLoader(test_b_dataset_gold, batch_size=1, shuffle=False, num_workers=5, pin_memory=True)


pseudo = UNet(1, 1)
refine_net = UNet(1, 1)
# UNet(1, 1, norm_layer=nn.BatchNorm2d, affine=True)
net = UNet(1, 1, domain_specific=True)

pseudo.cuda()
refine_net.cuda()
net.cuda()

# list optimisers here...
# single optimiser variant 1

optimiser_ps = optim.Adam(pseudo.parameters(), lr=learning_rate)
optimiser_ref = optim.Adam(refine_net.parameters(), lr=learning_rate*10)
optimiser_net = optim.Adam(net.parameters(), lr=learning_rate)

scheduler_ps = None # StepLR(optimiser_ps, step_size=1, gamma=0.5)
scheduler_net = None # StepLR(optimiser_net, step_size=5, gamma=0.8)

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


for i in range(epochs_pseudo):
    # train pseudolabeller here
    t_dice, t_loss = train_net(pseudo, train_a_loader, 'A', 2, 1, i, criterion_seg, 
                               is_train=True, opt=optimiser_ps, scheduler=scheduler_ps, name='Pseudo train a')
    v_dice, v_loss = train_net(pseudo, val_a_loader, 'A', 2, 1, i, criterion_seg, 
                               is_train=False, name='Pseudo val a')
    v_b_train, v_b_loss = train_net(pseudo, val_b_loader, 'A', 2, 1, i, criterion_seg, 
                                    is_train=False, name='Pseudo val b')
    
    pseudo_train_a_dice.append(t_dice)
    pseudo_train_a_loss.append(t_loss)
    pseudo_val_a_dice.append(v_dice)
    pseudo_val_a_loss.append(v_loss)
    pseudo_val_b_dice.append(v_b_train)
    pseudo_val_b_loss.append(v_b_loss)
    
    # train refine net here! Dont need to keep the losses!
    train_net(refine_net, train_a_loader, 'A', 3, 1, i, criterion_reg, 
                               is_train=True, opt=optimiser_ref, name='Refine train a')
    
    train_net(refine_net, val_a_loader, 'A', 3, 1, i, criterion_reg, 
                               is_train=False, name='Refine val a')


pre_ssl_dice_no_ref, pre_ssl_loss_no_ref = train_net(pseudo, test_b_loader, 'A', 2, 1, 0, criterion_seg, 
                                       is_train=False, name='pre ssl test b - no refine')

pre_ssl_dice_gold_no_ref, pre_ssl_loss_gold_no_ref = train_net(pseudo, test_b_loader_gold, 'A', 2, 1, 0, criterion_seg, 
                                                 is_train=False, name='pre ssl test b gold - no refine')
    
pre_ssl_dice, pre_ssl_loss = train_net(pseudo, test_b_loader, 'A', 2, 1, 0, criterion_seg, 
                                       refiner=refine_net, is_train=False, name='pre ssl test b')

pre_ssl_dice_gold, pre_ssl_loss_gold = train_net(pseudo, test_b_loader_gold, 'A', 2, 1, 0, criterion_seg, 
                                                 refiner=refine_net, is_train=False, name='pre ssl test b gold')


# for i in range(epochs_ssl):
#     for j, data in enumerate(tqdm.tqdm(train_a_b_loader)):
#         net.train()
        
#         optimiser_net.zero_grad()
        
#         image_a = data['A'][0].cuda()
#         target_a = data['A'][1].cuda()
        
#         image_b = data['B'][0].cuda()
#         edge_b = data['B'][2].cuda()

#         with torch.no_grad():
#             pseudo.eval()
#             refine_net.eval()
            
#             target = pseudo(edge_b).cuda()
#             target_ref = refine_net(target).cuda()
#             target_b = torch.round((target_ref + target) / 2).cuda()
                
#         net.set_domain(0)
        
#         pred_a = net(image_a)
#         crit_a = criterion_seg(pred_a, target_a)
#         logit_a = net.logits
        
#         net.set_domain(1)

#         pred_b = net(image_b)
#         crit_b = criterion_seg(pred_b, target_b)
#         logit_b = net.logits

#         loss = crit_a + crit_b + WassersteinLoss(sliced=True)(logit_a, logit_b)
        
#         loss.backward()
        
#         optimiser_net.step()
        
#     scheduler_net.step()
    
#     v_dice, v_loss = train_net(net, val_b_loader, 'A', 0, 1, i, criterion_seg, is_train=False,
#                                refiner=refine_net, name='Parent val b')


# for i in range(epochs_ssl):
#     # train main network here - with labels from pseudo and refine networks
    
#     # train domain A - no results for now
#     net.set_domain(0)
#     t_dice, t_loss = train_net(net, train_a_loader, 'A', 0, 1, i, criterion_seg, is_train=True, 
#                                opt=optimiser_net, refiner=refine_net, name='Parent train a')
    
#     v_dice, v_loss = train_net(net, val_a_loader, 'A', 0, 1, i, criterion_seg, is_train=False,
#                                refiner=refine_net, name='Parent val a')
    
#     # train domain B - keep the results
#     net.set_domain(1)
#     t_dice, t_loss = train_net(net, train_b_loader, 'A', 0, 1, i, criterion_seg, is_train=True, 
#                                opt=optimiser_net, labeller=pseudo, refiner=refine_net, name='Parent train b')
    
#     v_dice, v_loss = train_net(net, val_b_loader, 'A', 0, 1, i, criterion_seg, is_train=False,
#                                refiner=refine_net, name='Parent val b')
    
#     parent_train_b_dice.append(t_dice)
#     parent_train_b_loss.append(t_loss)
#     parent_val_b_dice.append(v_dice)
#     parent_val_b_loss.append(v_loss)

    
# final_dice, final_loss = train_net(net, test_b_loader, 'A', 0, 1, 0, criterion_seg, 
#                                    refiner=refine_net, is_train=False, name='final test b')

# final_dice_gold, final_loss_gold = train_net(net, test_b_loader_gold, 'A', 0, 1, 0, criterion_seg, 
#                                    refiner=refine_net, is_train=False, name='final test b gold')


# save_path = os.path.join(result_path, project_name)

# if os.path.isdir(save_path) is False:
#     os.makedirs(save_path)
    
# all_results = {
#     "project_name": project_name,
#     "project_description": project_description,
#     "epochs_pseudo": epochs_pseudo,
#     "epochs_ssl": epochs_ssl,
#     "learning_rate": learning_rate,
#     "batch_size": batch_a_size,

#     "pseudo_train_a_dice": pseudo_train_a_dice,
#     "pseudo_train_a_loss": pseudo_train_a_loss,
#     "pseudo_val_a_dice": pseudo_val_a_dice,
#     "pseudo_val_a_loss": pseudo_val_a_loss,
#     "pseudo_val_b_dice": pseudo_val_b_dice,
#     "pseudo_val_b_loss": pseudo_val_b_loss,

#     "parent_train_b_dice": parent_train_b_dice,
#     "parent_train_b_loss": parent_train_b_loss,
#     "parent_val_b_dice": parent_val_b_dice,
#     "parent_val_b_loss": parent_val_b_loss,
    
#     "pre_ssl_dice_no_ref": pre_ssl_dice_no_ref,
#     "pre_ssl_loss_no_ref": pre_ssl_loss_no_ref,

#     "pre_ssl_dice_gold_no_ref": pre_ssl_dice_gold_no_ref,
#     "pre_ssl_loss_gold_no_ref": pre_ssl_loss_gold_no_ref,

#     "pre_ssl_dice": pre_ssl_dice,
#     "pre_ssl_loss": pre_ssl_loss,
    
#     "pre_ssl_dice_gold": pre_ssl_dice_gold,
#     "pre_ssl_loss_gold": pre_ssl_loss_gold,
    
#     "final_dice": final_dice, 
#     "final_loss": final_loss,
    
#     "final_dice_gold": final_dice_gold, 
#     "final_loss_gold": final_loss_gold
# }

# np.savez(os.path.join(save_path, 'params'), **all_results)
# torch.save(net.state_dict(), os.path.join(save_path, 'net'))
# torch.save(refine_net.state_dict(), os.path.join(save_path, 'refiner'))

