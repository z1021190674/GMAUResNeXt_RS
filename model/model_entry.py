"""
use select_model to select model from here
"""

from xml.etree.ElementInclude import include
import torch.nn as nn
from model.atturesnext_class_nlocal import AttUResNeXt_class_nlocal
from model.atturesnext_feat_nlocal import AttUResNeXt_feat_nlocal
from model.atturesnext_feat import AttUResNeXt_feat_v3, AttUResNeXt_feat_v2, AttUResNeXt_feat_v1
from model.atturesnext_class import AttUResNeXt_class_v2, AttUResNeXt_class_v1
from model.uresnext_nlocal import UResNeXt_nlocal
from model.uresnext import UResNeXt
from model.model_ablation import AttUResNeXt_class_v2_1, AttUResNeXt_class_v2_2, AttUResNeXt_class_v2_3  

from model.network import U_Net, AttU_Net, R2U_Net, R2AttU_Net

from model.DeepLabV3plus.deeplabv3plus import DeepLabV3plus
from model.FastFCN.fastfcn import FastFCN
from model.PSPNet.pspnet import PSPNet
# from model.DANet.danet import DANet
from model.unetpp import NestedUNet
import os


def select_model(args):
    """
    Description:
        select model according to args.model_type, using type2model dictionary
    """
    type2model = {
        'UResNeXt': UResNeXt(args, n_classes=args.num_classes),
        'UResNeXt_nlocal':UResNeXt_nlocal(args, n_classes=args.num_classes),
        'AttUResNeXt_feat_nlocal':AttUResNeXt_feat_nlocal(args, n_classes=args.num_classes),
        'AttUResNeXt_class_nlocal':AttUResNeXt_class_nlocal(args, n_classes=args.num_classes),
         'AttUResNeXt_class_v1': AttUResNeXt_class_v1(args, n_classes=args.num_classes),
        'AttUResNeXt_class_v2':AttUResNeXt_class_v2(args, n_classes=args.num_classes),
        'AttUResNeXt_feat_v2':AttUResNeXt_feat_v2(args, n_classes=args.num_classes),
        'AttUResNeXt_feat_v3':AttUResNeXt_feat_v3(args, n_classes=args.num_classes),
        'AttUResNeXt_feat_v1':AttUResNeXt_feat_v1(args, n_classes=args.num_classes),
        'R2U_Net':R2U_Net(output_ch=args.num_classes),
        'R2AttU_Net':R2AttU_Net(output_ch=args.num_classes),
        'U_Net':U_Net(output_ch=args.num_classes),
        'AttU_Net':AttU_Net(output_ch=args.num_classes),
        'DeepLabV3plus':DeepLabV3plus(class_num=args.num_classes),
        'FastFCN':FastFCN(class_num=args.num_classes),
        'PSPNet':PSPNet(class_num=args.num_classes),
        # 'DANet':DANet(nclass=args.num_classes),
        'NestedUNet':NestedUNet(None,in_channel=3,out_channel=args.num_classes),
        'AttUResNeXt_class_v2_1':AttUResNeXt_class_v2_1(args, n_classes=args.num_classes),
        'AttUResNeXt_class_v2_2':AttUResNeXt_class_v2_2(args, n_classes=args.num_classes),
        'AttUResNeXt_class_v2_3':AttUResNeXt_class_v2_3(args, n_classes=args.num_classes),
    }
    model = type2model[args.model]
    return model

def select_freeze_model(args):
    """
    Description:
        select the model, whose weight is used as the key of the dict of the freeze_dict in freeze_weight 
    """
    temp_args = args.model
    args.model = args.freeze_model
    model = select_model(args)
    args.model = temp_args
    return model




def equip_multi_gpu(model, args):
    """
    Description:
        using nn.DataParallel to implement muli-gpu-training;
        it may be replaced by other torch modules, like nn.distributeddataparallel ???
    """
    # model = nn.DataParallel(model, device_ids=args.gpus)
    if args.is_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
        model.cuda()
        if len(args.gpu_id.split(',')) > 1:
            model = nn.DataParallel(model)
    else:
        model.cpu()

    return model