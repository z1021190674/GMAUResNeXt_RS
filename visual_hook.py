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

### things left to do ###
# the selection of the att_map

### panel ###
# args.model -> choose the model
# args.load_model_path  -> choose the checkpoints of the model
# args.dataset -> choose dataset
# res in data = get_dataset(...) should be adjusted

res = 320 # resolution of the test image

# idx
idx_start = 150
idx_stop = 300

# for the visualization of GMAResUNeXt_class
class_idx = 3

img_file_dir = 'gid_uresnet' # the directory of the results to be saved !!!


def step_tta(args, model, img):
        shape = img.shape[2]
        # warp input
        if args.is_gpu:
            img = img.cuda()
            label = label.cuda()
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

### potsdam ###
# Impervious surfaces (RGB: 255, 255, 255)
# Building (RGB: 0, 0, 255)
# Low vegetation (RGB: 0, 255, 255)
# Tree (RGB: 0, 255, 0)
# Car (RGB: 255, 255, 0)
# Clutter/background (RGB: 255, 0, 0)
# label2color_dict = {
#                         "0": [255,255,255],
#                         "1": [0,0,255],
#                         "2": [0,255,255],
#                         "3": [0,255,0],
#                         "4": [255,255,0],
#                         "5": [255,0,0],
#                     }

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

### load the network ###
# parse args
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
# use hook to get the intermediate output

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook



# model.attblock1.register_forward_hook(get_activation('attblock1'))
# model.attblock2.register_forward_hook(get_activation('attblock2'))
# model.attblock3.register_forward_hook(get_activation('attblock3'))
# model.attblock4.register_forward_hook(get_activation('attblock4'))

### dataset ###
# choose the train set
dataset = get_dataset(args, res=res, is_train=True, is_eval=False)
for idx in range(idx_start, idx_stop):
    ### get data from Potsdam or GID Dataset ###
    img, label = dataset[idx]
    img = img.cuda()
    _, height, width = img.shape
    ### method of hook ###
    # forward prop to get the output
    output = model(img.unsqueeze(0))
    pred = torch.argmax(output, dim=1)
    pred_colored_label = label2color(pred.squeeze(0).cpu().numpy(), label2color_dict)

    # att_map1 = activation.pop('attblock1')
    # att_map2 = activation.pop('attblock2')
    # att_map3 = activation.pop('attblock3')
    # att_map4 = activation.pop('attblock4')
    # # att_map_final = (att_map1*model.a1.detach() + att_map2*model.a2.detach() + att_map3*model.a3.detach() + att_map4*model.a4.detach()) / 4
    # att_map_final = (att_map1*model.a1.detach() + att_map2*model.a2.detach() + att_map3*model.a3.detach() + att_map4*model.a4.detach()) / (model.a1.detach()+model.a2.detach()+model.a3.detach()+model.a4.detach())
    # # overlay the image
    # if 'class' in args.model:
    #     att_map1 = att_map1.cpu().squeeze(0).permute(1,2,0).numpy()[:,:, class_idx]
    #     att_map2 = att_map2.cpu().squeeze(0).permute(1,2,0).numpy()[:,:, class_idx]
    #     att_map3 = att_map3.cpu().squeeze(0).permute(1,2,0).numpy()[:,:, class_idx]
    #     att_map4 = att_map4.cpu().squeeze(0).permute(1,2,0).numpy()[:,:, class_idx]
    #     att_map_final = att_map_final.cpu().squeeze(0).permute(1,2,0).numpy()[:,:, class_idx]
    # else: # for GMAUResNeXt_feat
    #     att_map1 = att_map1.cpu().squeeze(0).permute(1,2,0).numpy()
    #     att_map2 = att_map2.cpu().squeeze(0).permute(1,2,0).numpy()
    #     att_map3 = att_map3.cpu().squeeze(0).permute(1,2,0).numpy()
    #     att_map4 = att_map4.cpu().squeeze(0).permute(1,2,0).numpy()
    #     att_map_final = att_map_final.cpu().squeeze(0).permute(1,2,0).numpy()
    
    img = img.permute((1,2,0)).cpu().numpy()
    colored_label = label2color(label.numpy(), label2color_dict)

    # att_map1 *= 256
    # att_map1 = att_map1.astype(np.uint8)
    # heatmap1 = cv2.applyColorMap(cv2.resize(att_map1, (width, height)), cv2.COLORMAP_JET)[:,:,[2,1,0]]
    # result1 = heatmap1 * 0.3 + img * 0.5

    # att_map2 *= 256
    # att_map2 = att_map2.astype(np.uint8)
    # heatmap2 = cv2.applyColorMap(cv2.resize(att_map2, (width, height)), cv2.COLORMAP_JET)[:,:,[2,1,0]]
    # result2 = heatmap2 * 0.3 + img * 0.5
    
    # att_map3 *= 256
    # att_map3 = att_map3.astype(np.uint8)
    # heatmap3 = cv2.applyColorMap(cv2.resize(att_map3, (width, height)), cv2.COLORMAP_JET)[:,:,[2,1,0]]
    # result3 = heatmap3 * 0.3 + img * 0.5

    # att_map4 *= 256
    # att_map4 = att_map4.astype(np.uint8)
    # heatmap4 = cv2.applyColorMap(cv2.resize(att_map4, (width, height)), cv2.COLORMAP_JET)[:,:,[2,1,0]]
    # result4 = heatmap4 * 0.3 + img * 0.5

    # att_map_final *= 256
    # att_map_final = att_map_final.astype(np.uint8)
    # heatmap_final = cv2.applyColorMap(cv2.resize(att_map_final, (width, height)), cv2.COLORMAP_JET)[:,:,[2,1,0]]
    # result_final = heatmap_final * 0.3 + img * 0.5

    # result = np.concatenate((img, colored_label, pred_colored_label, result1, result2, result3, result4, result_final), axis=1)
    # result = np.concatenate((img, colored_label, pred_colored_label), axis=1)
    result = pred_colored_label


    ### render the output according to the idx ###
    img_file = os.path.join(img_file_dir,dataset.img_list[idx].replace('.png','_CAM.png').split('/')[-1])
    cv2.imwrite(img_file.replace('.png','id'+str(idx))+'_CAM.jpg', result[:,:,[2,1,0]])
    activation = {}