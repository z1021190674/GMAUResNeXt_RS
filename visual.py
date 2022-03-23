from torchvision.models._utils import IntermediateLayerGetter

from model.model_entry import select_model, equip_multi_gpu
from options import parse_common_args, parse_test_args
from data.data_entry import get_dataset

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
idx_start = 0
idx_stop = 2

img_file_dir = 'visual' # the directory of the results to be saved !!!


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
# check the name for the attention(gate) map !!!
# return_layers = {'attblock1': 'att_1', 'attblock2': 'att_2', 'attblock3': 'att_3', 'attblock4': 'att_4',}
# model_int = IntermediateLayerGetter(model, return_layers)
model.cuda()
# model_int.cuda()
# model_int.eval()

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


### dataset ###
dataset = get_dataset(args, res=res, is_train=True, is_eval=False)
for idx in range(idx_start, idx_stop):
    ### get data from Potsdam or GID Dataset ###
    img, _ = dataset[idx]
    img = img.cuda()
    ### get the intermediateoutput of the network (the attention(gate) map) ###
    height, width, _ = img.shape
    ### method of intermediatelayergetter ###
    # att_map1 = model_int(img.unsqueeze(0))['att_1'] # the att_map to be got !!!
    model.attblock1.register_forward_hook(get_activation('attblock1'))
    output = model(img.unsqueeze(0))
    att_map1 = activation['attblock1']

    # overlay the image

    

    
    att_map1 = att_map1.cpu().squeeze(0).permute(1,2,0).numpy()[:,:,0]
    att_map1 *= 256
    att_map1 = att_map1.astype(np.uint8)
    heatmap1 = cv2.applyColorMap(cv2.resize(att_map1, (width, height)), cv2.COLORMAP_JET)
    result1 = heatmap1 * 0.3 + img * 0.5

    ### render the output ###
    
    img_file = os.path.join(img_file_dir,dataset.img_list[idx].replace('.png','_CAM.png'))
    cv2.imwrite(img_file.replace('.jpg','id'+str(idx))+'_CAM.jpg', result1)