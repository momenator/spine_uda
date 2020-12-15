"""UNet-based architectures """

import torch
import torch.nn as nn
from .modules import Up, Down, DoubleConv, conv_bn, WaveletUp, WaveletDown


class UNet(nn.Module):
    """Classic UNet
    """
    
    def __init__(self, 
                 n_channels, 
                 n_classes, 
                 bilinear=True,
                 norm_layer=nn.InstanceNorm2d, 
                 track_running_stats=False, 
                 affine=False, 
                 multiple_output=False, 
                 domain_specific=False):
        """
        Args:
            n_channels (int): input channels
            n_classes (int): number of classes
            bilinear (int): to use bilinear interpolation or not
        """
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.multiple_output = multiple_output
        self.domain_specific = domain_specific
        self.logits = None

        self.inc = DoubleConv(n_channels, 64, norm_layer=norm_layer, track_running_stats=track_running_stats, affine=affine, domain_specific=domain_specific)
        self.down1 = Down(64, 128, False, norm_layer=norm_layer, track_running_stats=track_running_stats, affine=affine, domain_specific=domain_specific)
        self.down2 = Down(128, 256, False, norm_layer=norm_layer, track_running_stats=track_running_stats, affine=affine, domain_specific=domain_specific)
        self.down3 = Down(256, 512, False, norm_layer=norm_layer, track_running_stats=track_running_stats, affine=affine, domain_specific=domain_specific)
        self.down4 = Down(512, 512, False, norm_layer=norm_layer, track_running_stats=track_running_stats, affine=affine, domain_specific=domain_specific)
                
        self.up1 = Up(1024, 256, False, bilinear, norm_layer=norm_layer, track_running_stats=track_running_stats, affine=affine, domain_specific=domain_specific)   
        self.up2 = Up(512, 128, False, bilinear, norm_layer=norm_layer, track_running_stats=track_running_stats, affine=affine, domain_specific=domain_specific)
        self.up3 = Up(256, 64, False, bilinear, norm_layer=norm_layer, track_running_stats=track_running_stats, affine=affine, domain_specific=domain_specific)
        self.up4 = Up(128, 64, False, bilinear, norm_layer=norm_layer, track_running_stats=track_running_stats, affine=affine, domain_specific=domain_specific)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
        
        if n_classes > 1:
            self.final = nn.Softmax()
        else:
            self.final = nn.Sigmoid()
        
        # multiple output for shape aware segmentation
        # TODO - cleaner way of doing this?
        if multiple_output:
            self.final_contour = nn.Sigmoid()
            self.final_edt = nn.Sigmoid()
        
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        logits = self.outc(x)
        self.logits = logits
        
        if self.multiple_output:
            return self.final(logits), self.final_contour(logits), self.final_edt(logits), logits
        
        return self.final(logits)
    
    
    def downsample(self, x):
        """Forward pass - downsample
        Args:
            x (torch.Tensor): input tensor
        
        Returns:
            x1, x2, x3, x4, x5 (torch.Tensor): output of first double 
            conv (x1) and every downsampling modules (x2, x3, x4, x5)
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x1, x2, x3, x4, x5
            
    
    def upsample(self, x1, x2, x3, x4, x5):
        """Forward pass - upsample
        Args:
            x1, x2, x3, x4, x5 (torch.Tensor): output tensor from downsampling
            x1 - highest, x5 - bottleneck layer, x2-4 in-between layers
        
        """
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        logits = self.outc(x)
        
        if self.multiple_output:
            return self.final(logits), self.final_contour(logits), self.final_edt(logits), logits
        
        return self.final(logits)
    
    
    def set_domain(self, domain):
        self.domain = domain
        for module in self.modules():
            module.domain = domain
        return self
    

class UNetDense(nn.Module):
    """Unet with dense block as per the QuickNAT implementation
    It's separated for now for the sake of readability
    """
    
    def __init__(self, 
                 n_channels, 
                 n_classes, 
                 inner_channels=64, 
                 norm_layer=nn.InstanceNorm2d, 
                 track_running_stats=False, 
                 affine=False):
        """
        Args:
            n_channels (int): input channels
            n_classes (int): number of classes
            bilinear (int): to use bilinear interpolation or not
        """
        super(UNetDense, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        # self.inner_channels = inner_channels

        self.inc = DoubleConv(n_channels, inner_channels, norm_layer=norm_layer, track_running_stats=track_running_stats, affine=affine)
        self.down1 = Down(inner_channels, inner_channels, False, True, True, norm_layer=norm_layer, track_running_stats=track_running_stats, affine=affine)
        self.down2 = Down(inner_channels, inner_channels, False, True, True, norm_layer=norm_layer, track_running_stats=track_running_stats, affine=affine)
        self.down3 = Down(inner_channels, inner_channels, False, True, True, norm_layer=norm_layer, track_running_stats=track_running_stats, affine=affine)
        self.down4 = Down(inner_channels, inner_channels, False, True, True, norm_layer=norm_layer, track_running_stats=track_running_stats, affine=affine)
        
        self.bottleneck = conv_bn(inner_channels, inner_channels, 5, 1, 2, norm_layer=norm_layer, track_running_stats=track_running_stats, affine=affine)
        
        self.up1 = Up(inner_channels * 2, inner_channels, False, False, True, True, norm_layer=norm_layer, track_running_stats=track_running_stats, affine=affine)        
        self.up2 = Up(inner_channels * 2, inner_channels, False, False, True, True, norm_layer=norm_layer, track_running_stats=track_running_stats, affine=affine)
        self.up3 = Up(inner_channels * 2, inner_channels, False, False, True, True, norm_layer=norm_layer, track_running_stats=track_running_stats, affine=affine)
        self.up4 = Up(inner_channels * 2, inner_channels, False, False, True, True, norm_layer=norm_layer, track_running_stats=track_running_stats, affine=affine)
        self.outc = nn.Conv2d(inner_channels, n_classes, kernel_size=1)
        self.logits = None
        if n_classes > 1:
            self.final = nn.Softmax()
        else:
            self.final = nn.Sigmoid()
    
    def forward(self, x):
        x1 = self.inc(x)
        x2, idx2 = self.down1(x1)
        x3, idx3 = self.down2(x2)
        x4, idx4 = self.down3(x3)
        x5, idx5 = self.down4(x4)
        x5 = self.bottleneck(x5)
        
        x = self.up1(x5, x4, idx5)
        x = self.up2(x, x3, idx4)
        x = self.up3(x, x2, idx3)
        x = self.up4(x, x1, idx2)
        logits = self.outc(x)
        
        self.logits = logits
        
        # return logits
        return self.final(logits)
        
    def downsample(self, x):
        """Forward pass - downsample
        Args:
            x (torch.Tensor): input tensor
        
        Returns:
            x1, x2, x3, x4, x5 (torch.Tensor): output of first double 
            conv (x1) and every downsampling modules (x2, x3, x4, x5)
        """
        x1 = self.inc(x)
        x2, idx2 = self.down1(x1)
        x3, idx3 = self.down2(x2)
        x4, idx4 = self.down3(x3)
        x5, idx5 = self.down4(x4)
        x5 = self.bottleneck(x5)
        return x1, x2, x3, x4, x5, (idx2, idx3, idx4, idx5)
    
    def upsample(self, x1, x2, x3, x4, x5, idx_set):
        """Forward pass - upsample
        Args:
            x1, x2, x3, x4, x5 (torch.Tensor): output tensor from downsampling
            x1 - highest, x5 - bottleneck layer, x2-4 in-between layers
        
        """
        x = self.up1(x5, x4, idx_set[3])
        x = self.up2(x, x3, idx_set[2])
        x = self.up3(x, x2, idx_set[1])
        x = self.up4(x, x1, idx_set[0])
        logits = self.outc(x)
        # return logits
        return self.final(logits)
    
    
class UNetWavelet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNetWavelet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        
        # Wavelet pooling is set in Down by default
        self.down1 = WaveletDown(64, 128)
        self.down2 = WaveletDown(128, 256)
        self.down3 = WaveletDown(256, 512)
        self.down4 = WaveletDown(512, 1024, expand_all=True)
        
        # each filters should have their own layer
        self.bottleneckll = DoubleConv(1024, 512, None, norm_layer=nn.BatchNorm2d, track_running_stats=True, affine=True)
        self.bottlenecklh = DoubleConv(1024, 512, None, norm_layer=nn.BatchNorm2d, track_running_stats=True, affine=True)
        self.bottleneckhl = DoubleConv(1024, 512, None, norm_layer=nn.BatchNorm2d, track_running_stats=True, affine=True)
        self.bottleneckhh = DoubleConv(1024, 512, None, norm_layer=nn.BatchNorm2d, track_running_stats=True, affine=True)

        # self.bottleneck_out = DoubleConv(1024, 512)
        
        # use wavelet unpooling here
        self.up1 = WaveletUp(512, 256)
        self.up2 = WaveletUp(256, 128)
        self.up3 = WaveletUp(128, 64)
        self.up4 = WaveletUp(64, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
        if n_classes > 1:
            self.final = nn.Softmax()
        else:
            self.final = nn.Sigmoid()

        
    def forward(self, x):
        x1 = self.inc(x)
        
        # I should get 5 feature maps of the same size (128 * 4)
        ll1, lh1, hl1, hh1 = self.down1(x1)
        
        # feature maps (256 * 4)
        ll2, lh2, hl2, hh2 = self.down2(ll1)
        
        # feature maps (512 * 4)
        ll3, lh3, hl3, hh3 = self.down3(ll2)
        
        # feature maps (1024 * 4)
        ll4, lh4, hl4, hh4 = self.down4(ll3)
                
        # bottleneck - 512
        ll4 = self.bottleneckll(ll4)
        lh4 = self.bottlenecklh(lh4)
        hl4 = self.bottleneckhl(hl4)
        hh4 = self.bottleneckhh(hh4)
        
        x = self.up1(ll3, ll4, lh4, hl4, hh4)
        x = self.up2(ll2, x, lh3, hl3, hh3)
        x = self.up3(ll1, x, lh2, hl2, hh2)
        x = self.up4(x1, x, lh1, hl1, hh1)
        
        logits = self.outc(x)
        logits = self.final(logits)
        
        return logits

    
class UNetWaveletV2(nn.Module):
    """UNet with wavelet pooling during testing
    Based on the WCT2 Network - Currently not in use...
    """
    
    def __init__(self, n_channels, n_classes, bilinear=True):
        """
        Args:
            n_channels (int): input channels
            n_classes (int): number of classes
            bilinear (int): to use bilinear interpolation or not
        """
        super(UNetWaveletV2, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128, wavelet_test=True)
        self.down2 = Down(128, 256, wavelet_test=True)
        self.down3 = Down(256, 512, wavelet_test=True)
        self.down4 = Down(512, 1024, wavelet_test=True)
                
        self.up1 = Up(1024, 512, True, bilinear)        
        self.up2 = Up(512, 256, True, bilinear)
        self.up3 = Up(256, 128, True, bilinear)
        self.up4 = Up(128, 64, True, bilinear)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
        
        # need to add redundant layer to make it fit
        self.conv1 = nn.Conv2d(1024, 512, kernel_size=1)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=1)
        self.conv3 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv4 = nn.Conv2d(128, 64, kernel_size=1)

        
    def downsample(self, x):
        """Forward pass - downsample
        Args:
            x (torch.Tensor): input tensor
        
        Returns:
            x1, x2, x3, x4, x5 (torch.Tensor): output of first double 
            conv (x1) and every downsampling modules (x2, x3, x4, x5)
        """
        if self.training:
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            return x1, x2, x3, x4, x5
        else:
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2[0])
            x4 = self.down3(x3[0])
            x5 = self.down4(x4[0])
            return x1, x2, x3, x4, x5 
            
    
    def upsample(self, x1, x2, x3, x4, x5):
        """Forward pass - upsample
        Args:
            x1, x2, x3, x4, x5 (torch.Tensor): output tensor from downsampling
            x1 - highest, x5 - bottleneck layer, x2-4 in-between layers
        
        """
        if self.training:
            x5 = self.conv1(x5)
            x = self.up1(x5, x4)
            
            x = self.conv2(x)
            x = self.up2(x, x3)
            
            x = self.conv3(x)
            x = self.up3(x, x2)
            
            x = self.conv4(x)
            x = self.up4(x, x1)
            
            logits = self.outc(x)

            return torch.sigmoid(logits)
        
        else:
            x = self.up1(x5[0], (x5[1], x5[2], x5[3]))
            x = self.up2(x, (x4[1], x4[2], x4[3]))
            x = self.up3(x, (x3[1], x3[2], x3[3]))
            x = self.up4(x, (x2[1], x2[2], x2[3]))
            logits = self.outc(x)

            return torch.sigmoid(logits)

        
class AutoEncoder(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(AutoEncoder, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        self.unet = nn.Sequential(
            DoubleConv(n_channels, 64, norm_layer=nn.BatchNorm2d, affine=True),
            nn.MaxPool2d(2),
            DoubleConv(64, 128, norm_layer=nn.BatchNorm2d, affine=True),
            nn.MaxPool2d(2),
            DoubleConv(128, 256, norm_layer=nn.BatchNorm2d, affine=True),
            nn.MaxPool2d(2),
            DoubleConv(256, 512, norm_layer=nn.BatchNorm2d, affine=True),
            nn.MaxPool2d(2),
            DoubleConv(512, 1024, norm_layer=nn.BatchNorm2d, affine=True),
            nn.ConvTranspose2d(1024, 1024, kernel_size=2, stride=2),
            DoubleConv(1024, 512, norm_layer=nn.BatchNorm2d, affine=True ),
            nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2),
            DoubleConv(512, 256, norm_layer=nn.BatchNorm2d, affine=True),
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
            DoubleConv(256, 128, norm_layer=nn.BatchNorm2d, affine=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            DoubleConv(64, n_classes, norm_layer=nn.BatchNorm2d, affine=True),
        )
        
        if n_classes > 1:
            self.final = nn.Softmax()
        else:
            self.final = nn.Sigmoid()
    
    def forward(self, x):
        logits = self.unet(x)
        return self.final(logits)
    
    