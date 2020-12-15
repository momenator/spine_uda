import torch
import torch.nn as nn
import os
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from .modules import WavePool, WaveUnpool, ImagePool, NLayerDiscriminator
from utils.metrics import compute_dice_metric
from utils.losses import DiceLoss
import numpy as np


class WaveEncoder(nn.Module):
    """Wavelet encoder in WCT2, only partial layers used"""
    
    def __init__(self):
        super(WaveEncoder, self).__init__()

        self.pad = nn.ReflectionPad2d(1)
        self.relu = nn.ReLU(inplace=True)

        self.conv0 = nn.Conv2d(3, 3, 1, 1, 0)
        self.conv1_1 = nn.Conv2d(3, 64, 3, 1, 0)
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 0)
        self.pool1 = WavePool(64)

        self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 0)
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 0)
        self.pool2 = WavePool(128)

        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 0)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_4 = nn.Conv2d(256, 256, 3, 1, 0)
        self.pool3 = WavePool(256)
        
        self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 0)

    def forward(self, x, skips):
        """Wavelet encoding - only up to level 2
        Args:
            x (torch.Tensor): input to be encoded
            skips (dict): dictionary to contain LH, HL, HH filter responses
        
        Returns:
            LL (torch.Tensor): output of LL filters
            skips (dict): dictionary containing said filters
        """
        # level 1
        out = self.conv0(x)
        out = self.relu(self.conv1_1(self.pad(out)))
                
        # level 2
        out = self.relu(self.conv1_2(self.pad(out)))
        skips['conv1_2'] = out
        LL, LH, HL, HH = self.pool1(out)        
        skips['pool1'] = [LH, HL, HH]
        
        return LL, skips


class WaveDecoder(nn.Module):
    """Wavelet encoder in WCT2, only partial layers used"""
    def __init__(self):
        super(WaveDecoder, self).__init__()
        multiply_in = 5

        self.pad = nn.ReflectionPad2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.conv4_1 = nn.Conv2d(512, 256, 3, 1, 0)

        self.recon_block3 = WaveUnpool(256)
        
        self.conv3_4_2 = nn.Conv2d(256*multiply_in, 256, 3, 1, 0)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_1 = nn.Conv2d(256, 128, 3, 1, 0)

        self.recon_block2 = WaveUnpool(128)
        
        self.conv2_2_2 = nn.Conv2d(128*multiply_in, 128, 3, 1, 0)
        self.conv2_1 = nn.Conv2d(128, 64, 3, 1, 0)

        self.recon_block1 = WaveUnpool(64)
        self.conv1_2_2 = nn.Conv2d(64*multiply_in, 64, 3, 1, 0)
        self.conv1_1 = nn.Conv2d(64, 3, 3, 1, 0)


    def forward(self, x, skips):
        """Decoder - upsample from level 2
        Args:
            x (torch.Tensor): input to be encoded
            skips (dict): dictionary containing LH, HL, HH filter responses
        
        Returns:
            out (torch.Tensor): output of wavelet unpooling layer     
        """
        
        LH, HL, HH = skips['pool1']
        original = skips['conv1_2'] if 'conv1_2' in skips.keys() else None
        
        out = self.recon_block1(x, LH, HL, HH, original)
        return out


class WCT2Features(nn.Module):
    """WCT2 transform with fixed input and output channels and handpicked LL filters
    """
    def __init__(self, filters=None, model_path_encoder=None, model_path_decoder=None):
        super(WCT2Features, self).__init__()
        self.encoder = WaveEncoder().cuda()
        self.decoder = WaveDecoder().cuda()
        
        self.encoder.load_state_dict(
            torch.load(os.path.join(model_path_encoder),
                       map_location=lambda storage, loc: storage))
        self.decoder.load_state_dict(
            torch.load(os.path.join(model_path_decoder), 
                       map_location=lambda storage, loc: storage))
        
        self.filters = filters
        # self.tanh = nn.Tanh()
                
        # chosen channels
        # self.ll_filter_idx = [4,7,11,24,25,27]
        
        # Sparsest CT channels  [25, 54,16,22,61,4,8,27,7,3]
        # self.ll_filter_idx = [15,2,41,12,39,1,42,23,51,38]
        # self.ll_filter_idx = [14 ,15 ,45 ,19 ,39, 1 ,42 ,23 ,51, 38]
    
    def forward(self, x):
        """Get WCT2 LL filters
        Args:
            x (torch.Tensor): input tensor
        
        Returns:
            out (torch.Tensor): output LL filters
        """
        skips = {}
        out, skips = self.encoder(x, skips)
        out = self.decoder(out, skips)
        out = out[:,:64,:,:]
        if self.filters != None:
            out = torch.index_select(out, 1, torch.tensor(self.filters).cuda())
        return out
    
    
class WCT2GANUNet(nn.Module):
    """WCT2 GAN UNet all in one class"""
    
    def __init__(self, g, seg, n_channels, lr=0.0002):
        super(WCT2GANUNet, self).__init__()
        
        # generator
        self.g = g.cuda()
        
        # discriminator
        self.d = NLayerDiscriminator(input_nc=n_channels).cuda()
        
        # segmentor
        self.seg = seg.cuda()
        
        self.lr = lr
        
        # optimisers here
        self.g_optim = optim.Adam(self.g.parameters(), lr=self.lr)
        self.seg_optim = optim.Adam(self.seg.parameters(), lr=self.lr)
        # self.optim = optim.Adam(chain(self.g.parameters(), self.seg.parameters()), lr=self.lr)
        self.d_optim = optim.SGD(self.d.parameters(), lr=self.lr, momentum=0.5)
        
        self.criterion_gan = nn.BCELoss()
        self.pool = ImagePool()
    
    
    def criterion_seg(self, prediction, target):
        return nn.BCELoss()(prediction, target) + DiceLoss()(prediction, target)
    
    
    def forward_gen(self, x):
        return self.g(x)
    
    
    def forward_seg(self, x):
        out = self.forward_gen(x)
        a1, a2, a3, a4, a5 = self.seg.downsample(out)        
        seg = self.seg.upsample(a1, a2, a3, a4, a5)        
        return seg
        
        
    def get_target(self, pred, is_true=True):
        """Return target tensor with similar shape to pred"""
        if is_true == True and np.random.random() > 0.65:
            return torch.ones(pred.size(), requires_grad=False).cuda()
        return torch.zeros(pred.size(), requires_grad=False).cuda()
        
#         # occasionally give wrong labels
#         if is_true == True and np.random.random() + 0.3 > 0.5:
#             # use soft label for true [0.7, 1.2]
#             return (1.2 - 0.7) * torch.rand(pred.size(), requires_grad=False).cuda() + 0.7
        
#         # use soft label [0, 0.1] for false
#         return 0.1 * torch.rand(pred.size(), requires_grad=False).cuda()
            
    
    def set_requires_grad(self, net, requires_grad=False):
        for param in net.parameters():
            param.requires_grad=requires_grad

    
    def step(self, x_s, x_t, y_s):        
        # GAN loss - update discriminator and generator here
        # GAN loss - max log(D(x)) + log(1 - D(G(x)))
        
        # update d only
        self.d_optim.zero_grad()
        
        out_x_s = self.forward_gen(x_s)
        out_x_t = self.forward_gen(x_t)
        x_s_real = self.d(out_x_s)

        target_real = self.get_target(x_s_real)        
        loss_real = self.criterion_gan(x_s_real, target_real)
        loss_real.backward()
        
        # get generated feature maps from pool / replay for stability
        x_s_fake_map = (self.pool.query(out_x_t)).detach()
        x_s_fake = self.d(x_s_fake_map)
        
        target_fake = self.get_target(x_s_fake, is_true=False)
        loss_fake = self.criterion_gan(x_s_fake, target_fake)
        loss_fake.backward()
        self.d_optim.step()
        
        # update g - max D(G(X))
        self.g_optim.zero_grad()
        x_s_fake = self.d(x_s_fake_map)
        target_real = self.get_target(x_s_real)        
        loss_g = self.criterion_gan(x_s_fake, target_real)
        loss_g.backward()
        self.g_optim.step()
        
        # Segmentation loss
        self.set_requires_grad(self.g, requires_grad=False)
        # self.g_optim.zero_grad()
        self.seg_optim.zero_grad()
        
        out_seg = self.forward_seg(x_s)
        seg_loss = self.criterion_seg(out_seg, y_s)
        seg_loss.backward()        
        
        # self.g_optim.step()
        self.seg_optim.step()
        
        # calculate dice score for current batch
        dice_score = compute_dice_metric(torch.round(out_seg), y_s).item()
        
        # backward pass
        return seg_loss.item(), (loss_real + loss_fake).item(), dice_score
    
    
    def save(self, path):
        print('saving model...')
        if os.path.isdir(path) == False:
            os.makedirs(path)
        torch.save(self.g.state_dict(), os.path.join(path,'g.pth'))
        torch.save(self.d.state_dict(), os.path.join(path,'d.pth'))
        torch.save(self.seg.state_dict(), os.path.join(path,'seg.pth'))
        print('saving done!')
    