### test ###
import torchvision
import torch


# backbone = models.resnext101_32x8d(pretrained=False)
# return_layers = {'relu': 'feat1', 'layer1': 'feat2',\
#     'layer2': 'feat3', 'layer3': 'feat4', 'layer4': 'feat5'}
# resnext = torchvision.models._utils.IntermediateLayerGetter(backbone, return_layers)
# out = resnext(torch.rand(1, 3, 256, 256))
# print([(k, v.shape) for k, v in out.items()])

# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
# # backbone.to(device)
# # summary(backbone,(3,256,256))

# # for name, module in backbone.named_modules():
# #     print(name)



### building test ###
import torch.nn as nn
import torchvision
import torch.nn.functional as f
import torch
### import for test ###
import torchvision.models as models
from torchsummary import summary

def weights_init(init_type='gaussian'):
    """
    from https://github.com/naoto0804/pytorch-inpainting-with-partial-conv/blob/master/net.py
    """
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    return init_fun

class PartialConv(nn.Module):
    """
    from https://github.com/naoto0804/pytorch-inpainting-with-partial-conv/blob/master/net.py
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups, bias)
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                   stride, padding, dilation, groups, False)
        self.input_conv.apply(weights_init('kaiming'))

        torch.nn.init.constant_(self.mask_conv.weight, 1.0)

        # mask is not updated
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, input):
        # http://masc.cs.gmu.edu/wiki/partialconv
        # C(X) = W^T * X + b, C(0) = b, D(M) = 1 * M + 0 = sum(M)
        # W^T* (M .* X) / sum(M) + b = [C(M .* X) – C(0)] / D(M) + C(0)

        mask = torch.ones_like(input)
        output = self.input_conv(input)
        if self.input_conv.bias is not None:
            output_bias = self.input_conv.bias.view(1, -1, 1, 1).expand_as(
                output)
        else:
            output_bias = torch.zeros_like(output)

        with torch.no_grad():
            output_mask = self.mask_conv(mask)

        no_update_holes = output_mask == 0
        mask_sum = output_mask.masked_fill_(no_update_holes, 1.0)

        output_pre = (output - output_bias) / mask_sum + output_bias
        output = output_pre.masked_fill_(no_update_holes, 0.0)

        return output

class AttResNeXt(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.conv1 = PartialConv(2048,2048,3)

    
    def forward(self, in_encoder):
        x = f.interpolate(in_encoder['feat5'], scale_factor=2, mode='bilinear')
        x = self.conv1(x)

        return x

backbone = models.resnext101_32x8d(pretrained=False)
return_layers = {'relu': 'feat1', 'layer1': 'feat2',\
    'layer2': 'feat3', 'layer3': 'feat4', 'layer4': 'feat5'}
resnext = torchvision.models._utils.IntermediateLayerGetter(backbone, return_layers)
attnet = AttResNeXt()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0

attnet.to(device), resnext.to(device)


out = resnext(torch.rand(1, 3, 256, 256).to(device))
out = attnet(out)

print(out.shape)

# print([(k, v.shape) for k, v in out.items()])

