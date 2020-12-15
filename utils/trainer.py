import torch
import numpy as np
import tqdm
from .metrics import compute_dice_metric
from .losses import entropy_loss


def threshold_prediction(img, threshold=0.8):
    return (img > threshold).float()


def train_segmentation_net(net, 
                           loader, 
                           data_key, 
                           image_idx, 
                           target_idx, 
                           criterion,
                           opt=None,
                           epoch=0, 
                           is_train=True, 
                           labeller=None,
                           threshold=0.8,
                           name='train', 
                           verbose=True):
    """Train or evaluate a segmentation network"""
    
    dice_scores = []
    losses = []
    
    for i, data in enumerate(tqdm.tqdm(loader)):                
        image = data[data_key][image_idx].cuda()
        
        if labeller == None:
            target = data[data_key][target_idx].cuda()
        else:
            labeller.eval()
            with torch.no_grad():
                # labeller should be fed edge images!
                # edge is always 2
                edge = data[data_key][2].cuda()
                edge_label = threshold_prediction(labeller(edge), threshold)
                target = torch.round(edge_label).detach().cuda()
                
        if is_train:
            net.train()
            opt.zero_grad()
            pred = net(image)
            loss = criterion(pred, target)
            pred = torch.round(pred)
            loss.backward()
            opt.step()
            loss = loss.item()
        else:
            net.eval()
            with torch.no_grad():
                pred = net(image)
                pred = torch.round(pred).cuda()
                loss = criterion(pred, target).item()
                pred = torch.round(pred)
        
        dice_score = compute_dice_metric(pred, target).item()
        dice_scores.append(dice_score)
        losses.append(loss)
                                
    # print dice scores here!
    mean_dice = np.mean(dice_scores)
    mean_loss = np.mean(losses)
    
    if verbose:
        print('{} - epoch {} - Avg dice: {}, Avg loss: {}'.format(name, epoch, mean_dice, mean_loss))
            
    # return none for the time being
    return mean_dice, mean_loss


def train_mcd(net, 
              loader, 
              image_idx, 
              target_idx,
              crit_cls,
              crit_d,
              opt_enc, 
              opt_dec,
              K=2,
              epoch=0, 
              verbose=True, 
              use_entropy=False):
    
    """Train MCD framework"""
    
    dice_scores = []
    
    for i, data in enumerate(tqdm.tqdm(loader)):                
        image_a = data['A'][image_idx].cuda()
        target_a = data['A'][target_idx].cuda()
        image_b = data['B'][image_idx].cuda()
        
        net.train()
        
        # Step 1 - train enc + decoder on domain A
        opt_enc.zero_grad()
        opt_dec.zero_grad()

        pred_a = net(image_a)
        seg_loss_1 = crit_cls(pred_a, target_a)

        if use_entropy:
            pred_b = net(image_b)
            ent_a = entropy_loss(pred_a)
            ent_b = entropy_loss(pred_b)
            seg_loss_entropy = seg_loss_1 + 0.001 * (ent_a + ent_b)
            seg_loss_entropy.backward()
        else:
            seg_loss_1 = crit_cls(pred_a, target_a)
            seg_loss_1.backward()
        
        opt_enc.step()
        opt_dec.step()
        
        # Step 2 - train dec + maximise discrepancy on domain A
        opt_dec.zero_grad()
        pred_a = net(image_a)
        seg_loss_2 = crit_cls(pred_a, target_a)
        combined_loss_2 = (2 * seg_loss_2) - crit_d(net.logits, net.logits)
        combined_loss_2.backward()
        opt_dec.step()
        
        # Step 3 - train enc + minimise discrepancy on domain B
        for i in range(K):
            opt_enc.zero_grad()
            net(image_b)
            loss_3 = crit_d(net.logits, net.logits)
            loss_3.backward()
            opt_enc.step()
        
        with torch.no_grad():
            net.eval()
            pred_seg_a = net(image_a)        
            dice_score_a = compute_dice_metric(torch.round(pred_seg_a), target_a).item()
            dice_scores.append(dice_score_a)
    
    mean_dice = np.mean(dice_scores)
    
    if verbose:
        print('Train MCD - epoch {} - Avg dice: {}'.format(epoch, mean_dice))

    return mean_dice
    
    