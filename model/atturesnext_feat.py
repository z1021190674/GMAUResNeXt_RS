"""
uresnext_nlocal with global attention
"""
from model.block import *

import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.models as models


class AttUResNeXt_feat_v1(nn.Module):
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
        # self.up1 = nn.ConvTranspose2d(2048, 2048, 3, stride=2, padding=1, output_padding=1)
        # self.up2 = nn.ConvTranspose2d(1024, 1024, 3, stride=2, padding=1, output_padding=1)
        # self.up3 = nn.ConvTranspose2d(512, 512, 3, stride=2, padding=1, output_padding=1)
        # self.up4 = nn.ConvTranspose2d(256, 256, 3, stride=2, padding=1, output_padding=1)
        # self.up5 = nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1)

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
        self.attblock1 = AttBlock_v1(2048, 2048)
        self.attblock2 = AttBlock_v1(1024, 1024)
        self.attblock3 = AttBlock_v1(512, 512)
        self.attblock4 = AttBlock_v1(256, 256)
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
        shape = (int(img.shape[2]/2),int(img.shape[2]/2)) # for attblock 
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
        # att_sum = (self.att_params[0]*att1 + self.att_params[1]*att2 
        #         +  self.att_params[2]*att3 + self.att_params[3]*att4) / 4.0   #weighted attention
        # att_sum = (self.a1*att1 + self.a2*att2 +  self.a3*att3 + self.a4*att4) / 4.0   #weighted attention
        att_sum = (self.a1*att1 + self.a2*att2 +  self.a3*att3 + self.a4*att4) / (self.a1 + self.a2 + self.a3 + self.a4)
        
        
        x = self.up(x)
        x = self.conv4(x)
        x = torch.cat((e0, x), dim=1)
        x = self.dconv4(x)
        x = att_sum * x
        # level 6
        x = self.up(x)
        x = self.dconv5(x)
        x = self.conv5(x)
        
        x = F.log_softmax(x, dim=1)

        return x


class AttUResNeXt_feat_v2(nn.Module):
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
        self.attblock1 = AttBlock_v2(256 + 2048, 2048)
        self.attblock2 = AttBlock_v2(256 + 1024, 1024)
        self.attblock3 = AttBlock_v2(256 + 512, 512)
        self.attblock4 = AttBlock_v2(256 + 256, 256)
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
        att1 = self.attblock1(x, context, (e1.shape[2]*2, e1.shape[2]*2))
        # level 2
        # interpolation -- mini-batch x channels x [optional depth] x [optional height] x width.
        x = self.up(x)
        # torch.cuda.empty_cache()
        x = self.conv1(x)
        x = torch.cat((e3, x), dim=1)
        x = self.dconv1(x)
        att2 = self.attblock2(x, context, (e1.shape[2]*2, e1.shape[2]*2))
        # level 3
        x = self.up(x)
        # torch.cuda.empty_cache()
        x = self.conv2(x)
        x = torch.cat((e2, x), dim=1)
        x = self.dconv2(x)
        att3 = self.attblock3(x, context, (e1.shape[2]*2, e1.shape[2]*2))
        # level 4
        x = self.up(x)
        # torch.cuda.empty_cache()
        x = self.conv3(x)
        x = torch.cat((e1, x), dim=1)
        x = self.dconv3(x)
        att4 = self.attblock4(x, context, (e1.shape[2]*2, e1.shape[2]*2))
        # level 5
        # att_sum = (self.att_params[0]*att1 + self.att_params[1]*att2 
        #         +  self.att_params[2]*att3 + self.att_params[3]*att4) / 4.0   #weighted attention
        # att_sum = (self.a1*att1 + self.a2*att2 +  self.a3*att3 + self.a4*att4) / 4.0   #weighted attention
        att_sum = (self.a1*att1 + self.a2*att2 +  self.a3*att3 + self.a4*att4) /  (self.a1 + self.a2 + self.a3 + self.a4)
        
        
        x = self.up(x)
        # torch.cuda.empty_cache()
        x = self.conv4(x)
        x = torch.cat((e0, x), dim=1)
        x = self.dconv4(x)
        x = att_sum * x
        # level 6
        x = self.up(x)
        x = self.dconv5(x)
        x = self.conv5(x)
        
        x = F.log_softmax(x, dim=1)

        return x


class AttUResNeXt_feat_v3(nn.Module):
    """Decoder part of the UNet
    use the fusion of featuremap of each level to create signal g
    Parameters:
            n_classes -- number of the classes of the given dataset
    Tips:
        set align_corners = True for better performance of semantic segmentation (https://github.com/pytorch/vision/issues/1708)
    """
    def __init__(self, args, n_classes=6):
        super().__init__()
        self.n_classes = n_classes

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

        self.aconv1 = DoubleConv(2048, 64, 512)
        self.aconv2 = DoubleConv(1024, 64, 256)
        self.aconv3 = DoubleConv(512, 64, 128)
        self.aconv4 = DoubleConv(256, 64, 64)

        ### decoder ###
        self.attblock = AttBlock_v3(256, 64, 128)
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
        shape = (int(img.shape[2]/2),int(img.shape[2]/2)) # for attblock
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
        att1 = self.aconv1(x)
        # level 2
        # interpolation -- mini-batch x channels x [optional depth] x [optional height] x width.
        x = self.up(x)
        x = self.conv1(x)
        x = torch.cat((e3, x), dim=1)
        x = self.dconv1(x)
        att2 = self.aconv2(x)
        # level 3
        x = self.up(x)
        x = self.conv2(x)
        x = torch.cat((e2, x), dim=1)
        x = self.dconv2(x)
        att3 = self.aconv3(x)
        # level 4
        x = self.up(x)
        x = self.conv3(x)
        x = torch.cat((e1, x), dim=1)
        x = self.dconv3(x)
        att4 = self.aconv4(x)
        # level 5
        att1 = F.interpolate(att1, size=shape, mode='bilinear', align_corners=True)
        att2 = F.interpolate(att2, size=shape, mode='bilinear', align_corners=True)
        att3 = F.interpolate(att3, size=shape, mode='bilinear', align_corners=True)
        att4 = F.interpolate(att4, size=shape, mode='bilinear', align_corners=True)
        att =  torch.cat((att1, att2, att3, att4), dim=1)
        
        x = self.up(x)
        x = self.conv4(x)
        x = torch.cat((e0, x), dim=1)
        x = self.dconv4(x)
        x = self.attblock(att, x) * x
        # level 6
        x = self.up(x)
        x = self.dconv5(x)
        x = self.conv5(x)
        
        x = F.log_softmax(x, dim=1)

        return x

