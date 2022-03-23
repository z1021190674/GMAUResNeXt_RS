"""
Building blocks for atturesnext
"""


import torch.nn as nn
import torch.nn.functional as F
import torch

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, is_up=True):
        """
        when is_up is false, you should give the size of the output to upsample explicitly
        """
        super(UpConv,self).__init__(), 
        self.is_up = is_up
        self.bnconv = BnConv(in_channels, out_channels, 3, padding='same')
        if is_up:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self,x, out_size=(400,400)):
        if self.is_up:
            x = self.up(x)
        else:
            x = F.interpolate(x, size=out_size, mode='bilinear', align_corners=True)
        x = self.bnconv(x)
        return x

class BnConv(nn.Module):
    """
    (convolution => [BN] => ReLU)
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding='same'):
        super().__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=3,stride=1,padding=1,groups=in_channels),
                nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,stride=1,padding=0,groups=1),
                # nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
    def forward(self, x):
        return self.conv(x)

class DoubleConv(nn.Module):
    """
    (convolution => [BN] => ReLU) * 2
    """
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            # nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=3,stride=1,padding=1,groups=in_channels),
            # nn.Conv2d(in_channels=in_channels,out_channels=mid_channels,kernel_size=1,stride=1,padding=0,groups=1),
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding='same'),
            nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True),
            # nn.Conv2d(in_channels=mid_channels,out_channels=mid_channels,kernel_size=3,stride=1,padding=1,groups=mid_channels),
            # nn.Conv2d(in_channels=mid_channels,out_channels=out_channels,kernel_size=1,stride=1,padding=0,groups=1),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding='same'),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.double_conv(x)


class NLBlockND(nn.Module):
    def __init__(self,
                 in_channels,
                 inter_channels=None,
                 mode='embedded',
                 dimension=2,
                 bn_layer=True):
        """Implementation of Non-Local Block with 4 different pairwise functions but doesn't include subsampling trick
        from: https://github.com/tea1528/Non-Local-NN-Pytorch/blob/master/models/non_local.py
              https://github.com/AlexHex7/Non-local_pytorch/blob/master/lib/non_local_embedded_gaussian.py
        args:
            in_channels: original channel size (1024 in the paper)
            inter_channels: channel size inside the block if not specifed reduced to half (512 in the paper)
            mode: supports Gaussian, Embedded Gaussian, Dot Product, and Concatenation
            dimension: can be 1 (temporal), 2 (spatial), 3 (spatiotemporal)
            bn_layer: whether to add batch norm
        """
        super(NLBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
            raise ValueError(
                '`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`'
            )

        self.mode = mode
        self.dimension = dimension

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        # the channel size is reduced to half inside the block
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        # assign appropriate convolutional, max pool, and batch norm layers for different dimensions
        if dimension == 3:
            conv_nd = nn.Conv3d
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            bn = nn.BatchNorm1d

        # function g in the paper which goes through conv. with kernel size 1
        self.g = conv_nd(in_channels=self.in_channels,
                         out_channels=self.inter_channels,
                         kernel_size=1)

        # add BatchNorm layer after the last conv layer
        if bn_layer:
            self.W_z = nn.Sequential(
                conv_nd(in_channels=self.inter_channels,
                        out_channels=self.in_channels,
                        kernel_size=1), bn(self.in_channels))
            # from section 4.1 of the paper, initializing params of BN ensures that the initial state of non-local block is identity mapping
            nn.init.constant_(self.W_z[1].weight, 0)
            nn.init.constant_(self.W_z[1].bias, 0)
        else:
            self.W_z = conv_nd(in_channels=self.inter_channels,
                               out_channels=self.in_channels,
                               kernel_size=1)

            # from section 3.3 of the paper by initializing Wz to 0, this block can be inserted to any existing architecture
            nn.init.constant_(self.W_z.weight, 0)
            nn.init.constant_(self.W_z.bias, 0)

        # define theta and phi for all operations except gaussian
        if self.mode == "embedded" or self.mode == "dot" or self.mode == "concatenate":
            self.theta = conv_nd(in_channels=self.in_channels,
                                 out_channels=self.inter_channels,
                                 kernel_size=1)
            self.phi = conv_nd(in_channels=self.in_channels,
                               out_channels=self.inter_channels,
                               kernel_size=1)

        if self.mode == "concatenate":
            self.W_f = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels * 2,
                          out_channels=1,
                          kernel_size=1), nn.ReLU())

    def forward(self, x):
        """
        Params:
            x: (N, C, T, H, W) for dimension=3; (N, C, H, W) for dimension 2; (N, C, T) for dimension 1
        """

        batch_size = x.size(0)

        # (N, C, THW)
        # this reshaping and permutation is from the spacetime_nonlocal function in the original Caffe2 implementation
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        if self.mode == "gaussian":
            theta_x = x.view(batch_size, self.in_channels, -1)
            phi_x = x.view(batch_size, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "embedded" or self.mode == "dot":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "concatenate":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1,
                                         1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, 1, -1)

            h = theta_x.size(2)
            w = phi_x.size(3)
            theta_x = theta_x.repeat(1, 1, 1, w)
            phi_x = phi_x.repeat(1, 1, h, 1)

            concat = torch.cat([theta_x, phi_x], dim=1)
            f = self.W_f(concat)
            f = f.view(f.size(0), f.size(2), f.size(3))

        if self.mode == "gaussian" or self.mode == "embedded":
            f_div_C = F.softmax(f, dim=-1)
        elif self.mode == "dot" or self.mode == "concatenate":
            N = f.size(-1)  # number of position in x
            f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)

        # contiguous here just allocates contiguous chunk of memory
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])

        W_y = self.W_z(y)
        # residual connection
        z = W_y + x

        return z


class AttBlock_v1(nn.Module):
    """
    Description:
        whether using double conv after the interpolation?? -- the default is not
        using the feature of each level independently to generate the attention 
    Params:
        in_channels -- the input channel C of featuremap (N, C, H, W)
        out_channel -- the intermediate channel of the feature map, which is set the same as 
                    the input feature before concatenation
        shape -- shape=H=W, where the featuremap of the corresponding level is of shape (H*W)
    Tips:
        out_channels equals one, which is the default set, is for the feature attention
        out_channels equals num_classes for the class attention
    
    """
    def __init__(self, in_channels, mid_channels, out_channels=1):
        super().__init__()
        self.bnconv1 = DoubleConv(in_channels, mid_channels)
        self.conv1 = nn.Conv2d(mid_channels, out_channels, 3, padding='same')
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding='same')
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, shape=(400,400)):
        x = self.bnconv1(x)
        # generate the attention gate of the corresponding level(use the strategy of "Attention U-Net")
        x = self.conv1(x)
        x = F.interpolate(x, size=shape, mode='bilinear', align_corners=True)
        x = self.conv2(x)
        x = self.sigmoid(x)
        return x


class AttBlock_v2(nn.Module):
    """
    Description:
        convolve using context and featuremap of each level together -- attunet convolve using them separately(v3,v4)
        add convolution after interpolation 
    Params:
        in_channels -- the input channel C of featuremap (N, C, H, W)
        out_channel -- the intermediate channel of the feature map, which is set the same as 
                    the input feature before concatenation
        shape -- shape=H=W, where the output featuremap is of shape (H*W)
    Tips:
        out_channels equals one, which is the default set, is for the feature attention
        out_channels equals num_classes for the class attention
    
    """
    # def __init__(self, in_channels, mid_channels, out_channels=1):
    #     super().__init__()
    #     self.bnconv1 = BnConv(in_channels, mid_channels, 3)
    #     self.upconv1 = UpConv(mid_channels, out_channels, is_up=False)
    #     self.conv3 = nn.Conv2d(out_channels, out_channels, 3, padding='same')
    #     self.sigmoid = nn.Sigmoid()

    # def forward(self, feature, context, shape=(400,400)):
    #     # cast the featuremap into the internal feature space
    #     self.shape = feature.shape[2]
    #     context = context.expand(-1, -1, self.shape, self.shape)
    #     x = torch.cat((feature, context), dim=1)
    #     x = self.bnconv1(x)
    #     # generate the attention gate of the corresponding level(use the strategy of "Attention U-Net")
    #     x = self.upconv1(x, shape)
    #     # x = F.interpolate(x, size=shape, mode='bilinear', align_corners=True)
    #     x = self.sigmoid(self.conv3(x))
        
    #     return x
    def __init__(self, in_channels, mid_channels, out_channels=1):
        super().__init__()
        self.bnconv1 = BnConv(in_channels, mid_channels, 3)
        self.bnconv2 = BnConv(mid_channels, out_channels, 3)
        self.upconv1 = UpConv(out_channels, out_channels, is_up=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, feature, context, shape=(400,400)):
        # cast the featuremap into the internal feature space
        self.shape = feature.shape[2]
        context = context.expand(-1, -1, self.shape, self.shape)
        x = torch.cat((feature, context), dim=1)
        x = self.bnconv1(x)
        x = self.bnconv2(x)
        # generate the attention gate of the corresponding level(use the strategy of "Attention U-Net")
        x = self.upconv1(x, shape)
        x = self.sigmoid(x)
        
        return x


class AttBlock_v3(nn.Module):
    """
    Description:
        kernel_size equals 3
        context is generated from the last featuremap and of the same shape as the last featuremap
    Params:
        shape -- shape of the featuremap, a tuple like (h,w)
    """
    def __init__(self, F_g, F_l, F_int, out_channels=1):
        super(AttBlock_v3, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=3,stride=1,padding='same',bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, out_channels, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()
        )
                
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        att = self.psi(psi)

        return att

class AttBlock_v4(nn.Module):
    """
    Description:
        kernel_size equals 3
        context is generated from the last featuremap and of the same shape as the last featuremap
    Params:
        shape -- shape of the featuremap, a tuple like (h,w)
    """
    def __init__(self, F_g, F_l, F_int, out_channels=1):
        super(AttBlock_v3, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=3,stride=1,padding='same',bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, out_channels, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()
        )
        
        self.up2 = UpConv(F_int, F_int)
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x, shape):
        g = F.interpolate(g, size=shape, mode='bilinear', align_corners=True)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        att = self.psi(psi)

        return att