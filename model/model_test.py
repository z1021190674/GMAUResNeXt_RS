### used in model ###

# class AttUResNeXt(nn.Module):
#     """Decoder part of the UNet
#     Parameters:
#             n_classes -- number of the classes
#     Tips:
#         set align_corners = True for better performance of semantic segmentation (https://github.com/pytorch/vision/issues/1708)
#     """
#     def __init__(self, n_classes=5):
#         super().__init__()
#         self.n_classes = n_classes
#         self.gap = nn.AdaptiveAvgPool2d((1, 1))
#         # level 1
#         self.nlocal1 = NLBlockND(2048, inter_channels=1024)  # half the inter channels for computational efficiency
#         self.conv1 = nn.Conv2d(2048, 1024, 3, padding='same')
#         self.attblock1 = AttBlock(2048 + 2048, 2048, shape=8)
#         # level 2
#         self.nlocal2 = NLBlockND(1024, inter_channels=512)
#         self.dconv1 = DoubleConv(2048, 1024)
#         self.attblock2 = AttBlock(2048 + 1024, 1024, shape=16)
#         self.conv2 = nn.Conv2d(1024, 512, 3, padding='same')
#         # level 3
#         self.nlocal3 = NLBlockND(512, inter_channels=256)
#         self.dconv2 = DoubleConv(1024, 512)
#         self.attblock3 = AttBlock(2048 + 512, 512, shape=32)
#         self.conv3 = nn.Conv2d(512, 256, 3, padding='same')
#         # level 4
#         self.dconv3 = DoubleConv(512, 256)
#         self.attblock4 = AttBlock(2048 + 256, 256, shape=64)
#         self.conv4 = nn.Conv2d(256, 64, 3, padding='same')
#         # level 5
#         self.dconv4 = DoubleConv(128, 64)
#         # level 6
#         self.dconv5 = DoubleConv(64, 64)
#         self.conv5 = nn.Conv2d(64, self.n_classes, 3, padding='same')

#     def forward(self, in_encoder):
#         x = self.nlocal1(in_encoder['feat5'])
#         context = self.gap(x)
#         # level 1
#         att1 = self.attblock1(in_encoder['feat5'], context)
#         # level 2
#         # interpolation -- mini-batch x channels x [optional depth] x [optional height] x width.
#         x = F.interpolate(in_encoder['feat5'], size=(16,16), mode='bilinear', align_corners=True)
#         x = self.conv1(x)
#         x = torch.cat((in_encoder['feat4'], x), dim=1)
#         x = self.dconv1(x)
#         x = self.nlocal2(x)
#         att2 = self.attblock2(x, context)
#         # level 3
#         x = F.interpolate(x, size=(32,32), mode='bilinear',  align_corners=True)
#         x = self.conv2(x)
#         x = torch.cat((in_encoder['feat3'], x), dim=1)
#         x = self.dconv2(x)
#         x = self.nlocal3(x)
#         att3 = self.attblock3(x, context)
#         # level 4
#         x = F.interpolate(x, size=(64,64), mode='bilinear', align_corners=True)
#         x = self.conv3(x)
#         x = torch.cat((in_encoder['feat2'], x), dim=1)
#         x = self.dconv3(x)
#         att4 = self.attblock4(x, context)
#         # level 5
#         att_sum = (att1 + att2 + att3 + att4) / 4.0
#         x = F.interpolate(x, size=(128,128), mode='bilinear', align_corners=True)
#         x = self.conv4(x)
#         x = torch.cat((in_encoder['feat1'], x), dim=1)
#         x = self.dconv4(x)
#         x = att_sum * x
#         # level 6
#         x = F.interpolate(x, size=(256,256), mode='bilinear', align_corners=True)
#         x = self.dconv5(x)
#         x = self.conv5(x)
        

#         return x

if __name__ == '__main__':
    # return_layers = {'relu': 'feat1', 'layer1': 'feat2',\
    #     'layer2': 'feat3', 'layer3': 'feat4', 'layer4': 'feat5'}
    # resnext = torchvision.models._utils.IntermediateLayerGetter(
    #     backbone, return_layers)
    # attnet = AttUResNeXt(3)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # attnet.to(device), resnext.to(device)

    # out = resnext(torch.rand(1, 3, 256, 256).to(device))
    # out = attnet(out)
    pass
