"""
uresnext_nlocal with global attention
"""
from model.block import *

import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.models as models

class AttUResNeXt_class_v1(nn.Module):
    """Decoder part of the UNet
    Parameters:
            n_classes -- number of the classes of the given dataset
    Tips:
        set align_corners = True for better performance of semantic segmentation (https://github.com/pytorch/vision/issues/1708)
    """
    def __init__(self, args, n_classes=6):
        super().__init__()
        self.n_classes = n_classes

        # self.att_params = torch.nn.ParameterList([torch.nn.Parameter(torch.ones(1, dtype=torch.float32))
        #                                          for i in range(4)])
        self.a1 = torch.nn.Parameter(torch.tensor(1,dtype=torch.float32))
        self.a2 = torch.nn.Parameter(torch.tensor(1,dtype=torch.float32))
        self.a3 = torch.nn.Parameter(torch.tensor(1,dtype=torch.float32))
        self.a4 = torch.nn.Parameter(torch.tensor(1,dtype=torch.float32))

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)


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
        self.attblock1 = AttBlock_v1(2048, 2048, out_channels=n_classes)
        self.attblock2 = AttBlock_v1(1024, 1024, out_channels=n_classes)
        self.attblock3 = AttBlock_v1(512, 512, out_channels=n_classes)
        self.attblock4 = AttBlock_v1(256, 256, out_channels=n_classes)
        # level 1
        self.nlocal1 = NLBlockND(2048, inter_channels=1024)  # half the inter channels for computational efficiency
        self.conv1 = BnConv(2048, 1024, 3, padding='same')
        # level 2
        self.dconv1 = DoubleConv(2048, 1024)
        self.conv2 = BnConv(1024, 512, 3, padding='same')
        # level 3
        self.dconv2 = DoubleConv(1024, 512)
        self.conv3 = BnConv(512, 256, 3, padding='same')
        # level 4
        self.dconv3 = DoubleConv(512, 256)
        self.conv4 = BnConv(256, 64, 3, padding='same')
        # level 5
        self.dconv4 = DoubleConv(128, 64)
        # level 6
        self.dconv5 = DoubleConv(64, 64)
        self.conv5 = BnConv(64, self.n_classes, 3, padding='same')

    def forward(self, img):
        shape = (img.shape[2], img.shape[3]) # for attblock 
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
        att1 = self.attblock1(x, shape)
        # level 2
        # interpolation -- mini-batch x channels x [optional depth] x [optional height] x width.
        x = self.up(x)
        x = self.conv1(x)
        x = torch.cat((e3, x), dim=1)
        x = self.dconv1(x)
        att2 = self.attblock2(x, shape)
        # level 3
        x = self.up(x)
        x = self.conv2(x)
        x = torch.cat((e2, x), dim=1)
        x = self.dconv2(x)
        att3 = self.attblock3(x, shape)
        # level 4
        x = self.up(x)
        x = self.conv3(x)
        x = torch.cat((e1, x), dim=1)
        x = self.dconv3(x)
        att4 = self.attblock4(x, shape)
        # level 5        
        x = self.up(x)
        x = self.conv4(x)
        x = torch.cat((e0, x), dim=1)
        x = self.dconv4(x)
        
        # level 6
        x = self.up(x)
        x = self.dconv5(x)
        x = self.conv5(x)

        # att_sum = (self.a1*att1 + self.a2*att2 +  self.a3*att3 + self.a4*att4) / 4.0   #weighted attention
        att_sum = (self.a1*att1 + self.a2*att2 +  self.a3*att3 + self.a4*att4) / (self.a1 + self.a2 + self.a3 + self.a4)   #weighted attention
        x = att_sum * x
        x = F.log_softmax(x, dim=1)

        return x

class AttUResNeXt_class_v2(nn.Module):
    """Decoder part of the UNet
    use attblock_v1, however using the context
    Parameters:
            n_classes -- number of the classes of the given dataset
    Tips:
        set align_corners = True for better performance of semantic segmentation (https://github.com/pytorch/vision/issues/1708)
    """
    def __init__(self, args, n_classes=6):
        super().__init__()
        self.n_classes = n_classes

        self.a1 = torch.nn.Parameter(torch.tensor(1,dtype=torch.float32))
        self.a2 = torch.nn.Parameter(torch.tensor(1,dtype=torch.float32))
        self.a3 = torch.nn.Parameter(torch.tensor(1,dtype=torch.float32))
        self.a4 = torch.nn.Parameter(torch.tensor(1,dtype=torch.float32))

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

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
        self.attblock1 = AttBlock_v2(2048 + 256, 2048, out_channels=n_classes)
        self.attblock2 = AttBlock_v2(1024 + 256, 1024, out_channels=n_classes)
        self.attblock3 = AttBlock_v2(512 + 256, 512, out_channels=n_classes)
        self.attblock4 = AttBlock_v2(256 + 256, 256, out_channels=n_classes)
        # level 1
        self.nlocal1 = NLBlockND(2048, inter_channels=1024)  # half the inter channels for computational efficiency
        self.gconv1 = DoubleConv(2048,256,512)  # for decrease the channel of global context
        self.conv1 = BnConv(2048, 1024, 3, padding='same')
        # level 2
        self.dconv1 = DoubleConv(2048, 1024)
        self.conv2 = BnConv(1024, 512, 3, padding='same')
        # level 3
        self.dconv2 = DoubleConv(1024, 512)
        self.conv3 = BnConv(512, 256, 3, padding='same')
        # level 4
        self.dconv3 = DoubleConv(512, 256)
        self.conv4 = BnConv(256, 64, 3, padding='same')
        # level 5
        self.dconv4 = DoubleConv(128, 64)
        # level 6
        self.dconv5 = DoubleConv(64, 64)
        self.conv5 = BnConv(64, self.n_classes, 3, padding='same')

    def forward(self, img):
        shape = (img.shape[2], img.shape[2]) # for attblock 
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
        context = self.gap(self.gconv1(x))
        # level 1
        att1 = self.attblock1(x, context, shape)
        # level 2
        # interpolation -- mini-batch x channels x [optional depth] x [optional height] x width.
        x = self.up(x)
        x = self.conv1(x)
        x = torch.cat((e3, x), dim=1)
        x = self.dconv1(x)
        att2 = self.attblock2(x, context, shape)
        # level 3
        x = self.up(x)
        x = self.conv2(x)
        x = torch.cat((e2, x), dim=1)
        x = self.dconv2(x)
        att3 = self.attblock3(x, context, shape)
        # level 4
        x = self.up(x)
        x = self.conv3(x)
        x = torch.cat((e1, x), dim=1)
        x = self.dconv3(x)
        att4 = self.attblock4(x, context, shape)
        # level 5
        x = self.up(x)
        x = self.conv4(x)
        x = torch.cat((e0, x), dim=1)
        x = self.dconv4(x)
        # level 6
        x = self.up(x)
        x = self.dconv5(x)
        x = self.conv5(x)
        
        # att_sum = (self.a1*att1 + self.a2*att2 +  self.a3*att3 + self.a4*att4) / 4.0   #weighted attention
        att_sum = (self.a1*att1 + self.a2*att2 +  self.a3*att3 + self.a4*att4) / (self.a1 + self.a2 + self.a3 + self.a4)
        x = att_sum * x

        x = F.log_softmax(x, dim=1)

        return x


