import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DiceLoss(nn.Module):
    """Dice loss function"""
    
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 0.0001

    def forward(self, y_pred, y_true):
        """Compute dice coefficient as loss function
        Args:
            y_pred (torch.Tensor): predicted output
            y_true (torch.Tensor): ground truth
        
        Returns:
            score (torch.Float): dice loss
        """
        assert y_pred.size() == y_true.size()
        y_pred = y_pred.contiguous().view(-1)
        y_true = y_true.contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )
        return 1 - dsc
    

class WassersteinLoss(nn.Module):
    """Wasserstein (optimal transport) loss"""
    
    def __init__(self, sliced=False):
        """Init function. computes wasserstein loss
        
        Args:
            sliced (bool): flag to use sliced wasserstein distance
            absolute (bool): flaf to use absolute operation
        
        """
        
        super(WassersteinLoss, self).__init__()
        self.sliced = sliced

    def forward(self, source, target):
        """Compute wasserstein loss
        Args:
            source (torch.Tensor): input tensor
            target (torch.Tensor): ground truth
            
        Returns:
            wasserstein loss (torch.Tensor): wasserstein loss value
        """
        
        assert source.size() == target.size()
        
        if self.sliced:
            return sliced_wasserstein(source, target)
                
        return torch.mean(torch.abs(F.softmax(source, dim=1) - torch.softmax(target, dim=1)))
    
        
def sort_rows(m, n_rows):
    """sort N*M matrix by row
    
    Args:
        m (torch.Tensor): N*M matrix to be sorted
        n_rows (int): no of rows to be sorted
       
    Returns:
        sorted (torch.Tensor): N*M matrix with sorted row
    """
    m_T = m.transpose(1, 0)
    sorted_m = torch.topk(m_T, k=n_rows)[0]
    return sorted_m.transpose(1,0)


def sliced_wasserstein(s, t):
    """calculate sliced wasserstein distance between 2 tensors.
    step 1 - keep batch number
    step 2 - flatten x and y
    step 3 - perform projection to normal tensor with NxM dimension
    step 4 - matrix multiply x and y with projection
    step 5 - sort rows of x and y
    step 6 - compute wasserstein distance
    
    Args:
        s (torch.Tensor): source tensor input
        t (torch.Tensor): target tensor input
    
    Returns:
        score (float): sliced wasserstein distance
    """
    assert(s.size() == t.size())
    
    M = 128    
    N = s.size()[0]
    
    x = s.view(N, -1)
    y = t.view(N, -1)
    
    proj = torch.normal(0, 1, size=(x.size()[1], M))
    proj *= torch.sqrt(torch.sum(torch.pow(proj, 2), dim=0, keepdim=True))
        
    if torch.cuda.is_available():
        proj = proj.cuda()
    
    # should be NxM now...
    x = torch.matmul(x, proj)
    y = torch.matmul(y, proj)
    
    x = sort_rows(x, N)
    y = sort_rows(y, N)
        
    return torch.mean(torch.abs(F.softmax(x, dim=1) - F.softmax(y, dim=1)))


def entropy_loss(x):
    """Entropy loss as per equation 5 in FDA paper
    Adapted from: https://github.com/YanchaoYang/FDA and 
    https://github.com/valeoai/ADVENT/
    
    -torch.sum(torch.mul(v, torch.log2(v + 1e-30))) / (n * h * w * np.log2(c))

    
    Args:
        x (torch.Tensor): tensor output from segmentation network
    
    Returns:
        loss (torch.Float): entropy loss of x
    """
    
    n, c, h, w = x.size()
    ent = -torch.sum(torch.mul(F.softmax(x, dim=1), F.log_softmax(x, dim=1))) / (n * h * w * np.log2(c + 1e-2))
    
    ent = (ent ** 2.0 + 1e-6) ** 2
    return ent.mean()

    