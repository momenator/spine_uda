"""Contains common torch modules for various architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
import os
import functools
from itertools import chain
import random


"""Wavelet modules - taken from WCT2
"""

def get_wav(in_channels, pool=True):
    """Wavelet pooling / unpooling
    
    Args:
        in_channels (int): number of input channels
        pool (bool): flag to indicate pooling or unpooling
    
    Returns:
        LL, LH, HL, HH (nn.Conv2D / nn.ConvTranspose2d): Wavelet
        filters
    """
    harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H[0, 0] = -1 * harr_wav_H[0, 0]

    harr_wav_LL = np.transpose(harr_wav_L) * harr_wav_L
    harr_wav_LH = np.transpose(harr_wav_L) * harr_wav_H
    harr_wav_HL = np.transpose(harr_wav_H) * harr_wav_L
    harr_wav_HH = np.transpose(harr_wav_H) * harr_wav_H

    filter_LL = torch.from_numpy(harr_wav_LL).unsqueeze(0)
    filter_LH = torch.from_numpy(harr_wav_LH).unsqueeze(0)
    filter_HL = torch.from_numpy(harr_wav_HL).unsqueeze(0)
    filter_HH = torch.from_numpy(harr_wav_HH).unsqueeze(0)

    if pool:
        net = nn.Conv2d
    else:
        net = nn.ConvTranspose2d

    LL = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    LH = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HL = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HH = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)

    LL.weight.requires_grad = False
    LH.weight.requires_grad = False
    HL.weight.requires_grad = False
    HH.weight.requires_grad = False

    LL.weight.data = filter_LL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    LH.weight.data = filter_LH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    HL.weight.data = filter_HL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    HH.weight.data = filter_HH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)

    return LL, LH, HL, HH


class WavePool(nn.Module):
    """Wavelet pooling Module"""
    
    def __init__(self, in_channels):
        """Init function
        
        Args: 
            in_channels (int): number of input channels
        
        """
        super(WavePool, self).__init__()
        self.LL, self.LH, self.HL, self.HH = get_wav(in_channels)

    def forward(self, x):
        """Forward pass
        
        Args:
            x (torch.Tensor): Tensor to be pooled
        
        """
        return self.LL(x), self.LH(x), self.HL(x), self.HH(x)


class WaveUnpool(nn.Module):
    """Wavelet unpooling Module"""

    def __init__(self, in_channels, sum_pool=False):
        """Init function
        
        Args: 
            in_channels (int): number of input channels
        
        """
        super(WaveUnpool, self).__init__()
        self.in_channels = in_channels
        self.sum_pool = sum_pool
        self.LL, self.LH, self.HL, self.HH = get_wav(self.in_channels, pool=False)

    def forward(self, upsampled, LH, HL, HH, original=None):
        """Forward pass
        
        Args:
            upsampled (torch.Tensor): Upsampled tensor to be summed the other filter
            responses
            LH, HLM HH (torch.Tensor): Tensors taken from the skip connections
            
        Returns:
            tensor (torch.Tensor): Summed or concatenated sensor
        
        """
        if self.sum_pool:
            return self.LL(upsampled) + self.LH(LH) + self.HL(HL) + self.HH(HH)
        else:
            return torch.cat([self.LL(upsampled), self.LH(LH), self.HL(HL), 
                              self.HH(HH), original], dim=1)


"""Modules for UNet based architectures
"""

def conv_bn(in_channels, 
            out_channels, 
            kernel_size, 
            stride, 
            padding, 
            norm_layer,
            track_running_stats,
            affine, 
            domain_specific=False):
    """Convolution + norm layer
    
    Args:
        in_channels (int): input channels
        out_channels (int): output channels
        kernel_size (int): size of conv kernel
        stride (int): conv stride
        padding (int): conv padding
        norm_layer (torch.nn.Function): pytorch normalisation layer
        track_running_stats (boolean): flag to track running stats
        affine (boolean): flag to use affine params
    
    Returns:
        sequential layer (nn.Sequential) - conv + norm layer
    
    """
    
    if domain_specific:
        # use 2 domains and default settings of domain specific batch norm!
        normalisation = DomainSpecificNorm2d(out_channels, 2)
    else:
        normalisation = norm_layer(out_channels, affine=affine, track_running_stats=track_running_stats)
    
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, 
                  out_channels=out_channels, 
                  kernel_size=kernel_size, 
                  stride=stride,
                  padding=padding),
        normalisation
    )


class DenseBlock(nn.Module):
    """Dense block - interleaved convolution maps
    Specifically, we follow the QuickNAT implementation here...
    """
    def __init__(self, 
                 in_channels=64,
                 out_channels=64, 
                 norm_layer=nn.InstanceNorm2d, 
                 track_running_stats=False, 
                 affine=False):
        
        super().__init__()
        self.conv_bn1 = nn.Sequential(
            conv_bn(in_channels, 
                    in_channels, 
                    5, 
                    1, 
                    2, 
                    norm_layer=norm_layer, 
                    track_running_stats=track_running_stats, 
                    affine=affine),
            nn.ReLU(inplace=True),
        )
        
        self.conv_bn2 = nn.Sequential(
            conv_bn(in_channels * 2, 
                    in_channels, 
                    5, 
                    1, 
                    2, 
                    norm_layer=norm_layer, 
                    track_running_stats=track_running_stats, 
                    affine=affine),
            nn.ReLU(inplace=True),
        )
        
        self.conv_out = conv_bn(in_channels * 3, 
                                out_channels, 
                                1, 
                                1, 
                                0, 
                                norm_layer=norm_layer, 
                                track_running_stats=track_running_stats, 
                                affine=affine)
        
    def forward(self, x):
        out1 = self.conv_bn1(x)
        concat1 = torch.cat([x, out1], dim=1)
        out2 = self.conv_bn2(concat1)
        concat2 = torch.cat([x, out1, out2], dim=1)
        out = self.conv_out(concat2)
        return out

    

class DoubleConv(nn.Module):
    """Two conv_bn layers
    """

    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 mid_channels=None, 
                 norm_layer=nn.InstanceNorm2d, 
                 track_running_stats=False, 
                 affine=False, 
                 domain_specific=False):
        """
        Args:
            in_channels (int): number of incoming channels
            out_channels (int): number of outgoing channels
            mid_channels (int): number of channels in the middle
        """
        super().__init__()
        if mid_channels == None:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            conv_bn(in_channels, mid_channels, 3, 1, 1, norm_layer, track_running_stats, affine, domain_specific),
            nn.ReLU(inplace=True),
            conv_bn(mid_channels, out_channels, 3, 1, 1, norm_layer, track_running_stats, affine, domain_specific),
            nn.ReLU(inplace=True)
        )            

    def forward(self, x):
        """Forward pass
        Args:
            x (torch.Tensor): tensor
        
        """
        out = self.double_conv(x)
        return out
        
        
class Down(nn.Module):
    """Downsampling module with maxpool then double conv for UNet
    based architectures.
    """

    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 wavelet_test=False, 
                 dense=False, 
                 unpool=False, 
                 norm_layer=nn.InstanceNorm2d, 
                 track_running_stats=False, 
                 affine=False, 
                 domain_specific=False):
        """
        Args:
            in_channels (int): number of incoming channels
            out_channels (int): number of outgoing channels
        """
        super().__init__()
                
        self.unpool = unpool
        self.pool = nn.MaxPool2d(2, return_indices=unpool)
        
        if dense:
            # use the default 64 channels
            self.conv = DenseBlock(norm_layer=norm_layer, track_running_stats=track_running_stats, affine=affine)
        else:
            self.conv = DoubleConv(in_channels, out_channels, None, norm_layer, track_running_stats, affine, domain_specific)
        
        self.wavelet_test = wavelet_test
        if self.wavelet_test:
            self.wavepool = WavePool(in_channels)
                
    def forward(self, x):
        """Forward pass
        Args:
            x (torch.Tensor): tensor
        """
                
        if self.wavelet_test:
            if self.training:
                x = self.pool(x)
                x = self.conv(x)
                return x
            else:
                ll, lh, hl, hh = self.wavepool(x)
                ll = self.conv(ll)
                lh = self.conv(lh)
                hl = self.conv(hl)
                hh = self.conv(hh)
                return ll, lh, hl, hh
        else:
            if self.unpool:
                x, idx = self.pool(x)
                x = self.conv(x)
                return x, idx
            else:
                x = self.pool(x)
                return self.conv(x)
                    

class Up(nn.Module):
    """Upsampling module with upsample / conv tranpose then double conv for UNet
    based architectures
    """
    
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 wavelet_test=False, 
                 bilinear=True, 
                 dense=False, 
                 unpool=False,
                 norm_layer=nn.InstanceNorm2d, 
                 track_running_stats=False, 
                 affine=False, 
                 domain_specific=False):
        """
        Args:
            in_channels (int): number of incoming channels
            out_channels (int): number of outgoing channels
            bilinear (bool): flag to indicate the usage of upsampling / conv transpose
            layer
        """
        super().__init__()
        
        # if bilinear, use the normal convolutions to reduce the number of channels
        self.unpool = unpool
        
        if unpool:
            self.up = nn.MaxUnpool2d(2)
        elif bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        
        self.wavelet_test = wavelet_test
        if self.wavelet_test:
            self.waveunpool = WaveUnpool(in_channels)
            
        if dense:
            self.conv = DenseBlock(in_channels, norm_layer=norm_layer, track_running_stats=track_running_stats, affine=affine)
        else:
            self.conv = DoubleConv(in_channels, out_channels, None, norm_layer, track_running_stats, affine, domain_specific)
        
        
    def forward(self, x1, x2, idx=None):
        """Forward pass
        Args:
            x1 (torch.Tensor): tensor from lower level (to be upsampled)
            x2 (torch.Tensor): tensor from skip connection
        
        Returns:
            x (torch.Tensor): concatenated sensor (upsampled + skip connection)
        """
        
        if self.unpool and idx != None:
            up_x1 = self.up(x1, idx)
        else:
            up_x1 = self.up(x1)
        
        if self.training != True and self.wavelet_test == True:
            x3 = self.waveunpool(x1, x2[0], x2[1], x2[2])
            x = self.conv(x3)
            return x
            
        # There might be issues with dimension so pad them up here
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - up_x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - up_x1.size()[3]])

        # This is to make sure the dimensions are always even
        up_x1 = F.pad(up_x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, up_x1], dim=1)
        x = self.conv(x)
        return x

    
# Separate Up and Down module for wavelet pooling
class WaveletDown(nn.Module):
    """Downscaling with wavelet pooling followed by double convolution"""

    def __init__(self, in_channels, out_channels, expand_all=False):
        super().__init__()
        # TODO - can we make it such that we use pooling during training
        # then wavelet pooling during testing?
        
        # output channels is fixed - 5 * in_channels
        self.expand_all = expand_all
        self.pool = nn.MaxPool2d(2)
        self.wav_pool = WavePool(in_channels)
        self.conv = DoubleConv(in_channels, out_channels, 
                               None, norm_layer=nn.BatchNorm2d, track_running_stats=True, affine=True)
                
    def forward(self, x):
        # LH, HL, HH will be used for skip
        LL, LH, HL, HH = self.wav_pool(x)

        if self.expand_all:
            return self.conv(LL), self.conv(LH), self.conv(HL), self.conv(HH)
        return self.conv(LL), LH, HL, HH
        

class WaveletUp(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # this layer takes 5 input 
        self.up = WaveUnpool(in_channels)
        # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.conv = DoubleConv(in_channels*5, out_channels, mid_channels=in_channels,
                               norm_layer=nn.BatchNorm2d, track_running_stats=True, affine=True)
        
    def forward(self, original, LL, LH, HL, HH):
        up_original = self.up(LL, LH, HL, HH, original)
        x = self.conv(up_original)
        return x
    
    
class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator, used as discriminator for GAN 
    based training (real vs not real). Sigmoid layer is used at the end. Taken
    from CycleGAN repo.
    """

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Args:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        """Forward pass
        Args:
            input (torch.Tensor): input tensor
        
        Returns:
            result (torch.Tensor): one hot vector 
        
        """
        x = self.model(input)
        logits = F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)
        return self.sigmoid(logits)


class ImagePool():
    """Image pooling class to save generated GAN results for better stability"""
    
    def __init__(self, max_size=50):
        """
        Args:
            max_size (int): size of image pool
        """
        assert (max_size > 0)
        self.max_size = max_size
        self.data = []

    def query(self, data):
        """Push data to pool and return random image from pool
        Args:
            data (torch.Tensor): data to be pushed
        
        Returns:
            data (torch.Tensor): data to be returned
        
        """
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)

    
# Taken from https://github.com/wgchang/DSBN/blob/master/model/dsbn.py
class DomainSpecificNorm2d(nn.Module):
    _version = 2

    def __init__(self, num_features, num_classes, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super().__init__()
        self.dsbns = nn.ModuleList(
            [nn.BatchNorm2d(num_features, track_running_stats=track_running_stats, affine=affine) for _ in range(num_classes)])

    def reset_running_stats(self):
        for bn in self.bns:
            bn.reset_running_stats()

    def reset_parameters(self):
        for bn in self.bns:
            bn.reset_parameters()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))
        
    def _check_domain(self):
        if self.domain is None:
            raise AttributeError('Domain not set!')
        # TODO also make sure that domain is 0 or 1 for now!

    def forward(self, x):
        self._check_domain()
        self._check_input_dim(x)
        bn = self.dsbns[self.domain]
        return bn(x)

    
class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None
        