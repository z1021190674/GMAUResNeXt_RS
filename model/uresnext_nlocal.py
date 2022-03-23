"""
vanilla uresnext plus one nlocal block in each level 1,2,3
"""

from model.block import *

import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.models as models




class UResNeXt_nlocal(nn.Module):
    """Decoder part of the UNet
    Parameters:
            n_classes -- number of the classes
    Tips:
        set align_corners = True for better performance of semantic segmentation (https://github.com/pytorch/vision/issues/1708)
    """
    def __init__(self, args, n_classes=6):
        super().__init__()
        self.n_classes = n_classes
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        ### encoder ###
        resnext = models.resnext101_32x8d(pretrained=args.is_pretrained)
        self.firstconv = resnext.conv1
        self.firstbn = resnext.bn1
        self.firstrelu = resnext.relu
        self.firstmaxpool = resnext.maxpool
        self.encoder1 = resnext.layer1
        self.encoder2 = resnext.layer2
        self.encoder3 = resnext.layer3
        self.encoder4 = resnext.layer4

        ### decoder ###
        # level 1
        self.nlocal1 = NLBlockND(2048, inter_channels=1024)  # half the inter channels for computational efficiency
        self.conv1 = nn.Conv2d(2048, 1024, 3, padding='same')
        # level 2
        self.nlocal2 = NLBlockND(1024, inter_channels=512)
        self.dconv1 = DoubleConv(2048, 1024)
        self.conv2 = nn.Conv2d(1024, 512, 3, padding='same')
        # level 3
        self.nlocal3 = NLBlockND(512, inter_channels=256)
        self.dconv2 = DoubleConv(1024, 512)
        self.conv3 = nn.Conv2d(512, 256, 3, padding='same')
        # level 4
        self.dconv3 = DoubleConv(512, 256)
        self.conv4 = nn.Conv2d(256, 64, 3, padding='same')
        # level 5
        self.dconv4 = DoubleConv(128, 64)
        # level 6
        self.dconv5 = DoubleConv(64, 64)
        self.conv5 = nn.Conv2d(64, self.n_classes, 3, padding='same')

    def forward(self, img):
        ### encoder ###
        x1 = self.firstconv(img)
        x1 = self.firstbn(x1)
        e0 = self.firstrelu(x1)
        e1 = self.firstmaxpool(e0)

        e1 = self.encoder1(e1)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        ### decoder ###
        x = self.nlocal1(e4)
        # level 1
        # level 2
        # interpolation -- mini-batch x channels x [optional depth] x [optional height] x width.
        x = F.interpolate(x, size=(16,16), mode='bilinear', align_corners=True)
        x = self.conv1(x)
        x = torch.cat((e3, x), dim=1)
        x = self.dconv1(x)
        x = self.nlocal2(x)
        # level 3
        x = F.interpolate(x, size=(32,32), mode='bilinear',  align_corners=True)
        x = self.conv2(x)
        x = torch.cat((e2, x), dim=1)
        x = self.dconv2(x)
        x = self.nlocal3(x)
        # level 4
        x = F.interpolate(x, size=(64,64), mode='bilinear', align_corners=True)
        x = self.conv3(x)
        x = torch.cat((e1, x), dim=1)
        x = self.dconv3(x)
        # level 5
        x = F.interpolate(x, size=(128,128), mode='bilinear', align_corners=True)
        x = self.conv4(x)
        x = torch.cat((e0, x), dim=1)
        x = self.dconv4(x)
        # level 6
        x = F.interpolate(x, size=(256,256), mode='bilinear', align_corners=True)
        x = self.dconv5(x)
        x = self.conv5(x)
        
        x = F.log_softmax(x, dim=1)

        return x

if __name__ == '__main__':
    net = UResNeXt_nlocal()
    data = torch.rand(1, 3, 256, 256)
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(r'logs/resunext_nlocal')
    writer.add_graph(net, torch.rand(1,3,256,256))
    writer.close()
    pass