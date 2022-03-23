# from utils.torch_utils import freeze_weight, get_weight_dict, print_parameters
# from model.uresnext import UResNeXt
# from model.uresnext_nlocal import UResNeXt_nlocal
# import os

# from options import parse_common_args, parse_train_args
# import argparse
# parser = argparse.ArgumentParser()
# parser= parse_train_args(parser)
# args = parser.parse_args()
# args.is_pretrained = False


# # 创建的目录
# # path = "./logs/xx"

# # os.mkdir( path,755)

# model = UResNeXt(args)

# freeze_dict = get_weight_dict(model)

# model_freeze = UResNeXt_nlocal(args)
# freeze_weight(model_freeze, freeze_dict)

# print_parameters(model_freeze)


from model.atturesnext_feat import AttUResNeXt_feat_v1, AttUResNeXt_feat_v2, AttUResNeXt_feat_v3

import numpy as np


if __name__ == '__main__':
    # from model.torchsummary import summary
    # from model.atturesnext_class import AttUResNeXt_class_v2
    # from model.uresnext import UResNeXt

    # from options import parse_common_args, parse_train_args
    # import argparse

    # import torch
    # import torchvision.models as models

    # parser = argparse.ArgumentParser()
    # parser= parse_train_args(parser)
    # args = parser.parse_args()
    # args.is_pretrained = False



    # backbone = AttUResNeXt_class_v2(args)
    # # backbone = models.resnext101_32x8d(pretrained=args.is_pretrained)

    # data = torch.randn(1,3,1200,1200)
    # # y = backbone(data)
    # summary(backbone.cuda(), (3, 960, 960))

    # from torch.utils.tensorboard import SummaryWriter
    # import torch
    # writer = SummaryWriter(r'logs/model/AttUResNeXt_class_v2')
    # writer.add_graph(backbone, torch.rand(1,3,960,960).cuda())
    # writer.close()

    from torchvision.models._utils import IntermediateLayerGetter

    from model.model_entry import select_model, equip_multi_gpu
    from options import parse_test_args
    from data.data_entry import get_dataset
    from data.data_split.label_split_utils import label2color

    import torch
    from torchvision import transforms

    import cv2
    import numpy as np
    import os
    import argparse

    # label2color_dict = {
    #                     "0": [255,255,255],
    #                     "1": [0,0,255],
    #                     "2": [0,255,255],
    #                     "3": [0,255,0],
    #                     "4": [255,255,0],
    #                     "5": [255,0,0],
    #                 }

    label2color_dict = {
                                    "0": [200,0,0],
                                    "1": [250,0,150],
                                    "2": [200,150,150],
                                    "3": [250,150,150],
                                    "4": [0,200,0],
                                    "5": [150,250,0],
                                    "6": [150,200,150],
                                    "7": [200,0,200],
                                    "8": [150,0,250],
                                    "9": [150,150,250],
                                    "10": [250,200,0],
                                    "11": [200,200,0],
                                    "12": [0,0,200],
                                    "13": [0,150,200],
                                    "14": [0,200,250],
                                    "15": [0,0,0],
                                }


    res = 160
    image_path = "himage.png"
    # image_path = "top_potsdam_6_8_RGBIR_42.png"


    def step_tta(model, img):
        # hvflip
        img_tmp = transforms.functional.hflip(img)
        img_tmp = transforms.functional.vflip(img_tmp)
        prob1 = model(img_tmp)
        prob1 =  transforms.functional.vflip(prob1)
        prob1 =  transforms.functional.hflip(prob1)
        # h flip
        img_tmp = transforms.functional.hflip(img)
        prob2 = model(img_tmp)
        prob2 =  transforms.functional.hflip(prob2)
        # v filp
        img_tmp = transforms.functional.vflip(img)
        prob3 = model(img_tmp)
        prob3 =  transforms.functional.vflip(prob3)
        # no flip
        prob4 = model(img)
        # compute output
        prob = (prob1 + prob2 + prob3 + prob4) / 4.0
        return prob

    parser = argparse.ArgumentParser()
    parser= parse_test_args(parser)
    args = parser.parse_args()
    # load model
    model = select_model(args)
    # load state_dict
    checkpoint = torch.load(args.load_model_path)
    state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.cuda()
    model.eval()
    for j in range(5,11):
        res = j * 32
    # os.system('mkdir -p ' + str(res))
    # load image
        for i in range(1,6):
            tmp_path = image_path.replace('.png', str(i)+'.png')
            img = cv2.imread(tmp_path)[:,:,[2,1,0]]
            img = cv2.resize(img, (res,res))
            img = torch.from_numpy(img).float()
            img = img.permute((2, 0, 1)).unsqueeze(0)
            img = img.cuda()
        

            output = step_tta(model, img)
            pred = torch.argmax(output, dim=1)
            pred_colored_label = label2color(pred.squeeze(0).cpu().numpy(), label2color_dict)

            cv2.imwrite(str(res) + '/'+ tmp_path.replace('.png','_result.png'), pred_colored_label[:,:,[2,1,0]])

    # img = cv2.imread(image_path)[:,:,[2,1,0]]
    # img = torch.from_numpy(img).float()
    # img = img.permute((2, 0, 1)).unsqueeze(0)
    # img = img.cuda()
    # output = model(img)
    # pred = torch.argmax(output, dim=1)
    # pred_colored_label = label2color(pred.squeeze(0).cpu().numpy(), label2color_dict)
    # cv2.imwrite(image_path.replace('.png','_result.png'), pred_colored_label[:,:,[2,1,0]])