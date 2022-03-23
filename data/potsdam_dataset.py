"""
Potsdam dataset
"""



from torch.utils.data import Dataset
import numpy as np
import cv2
import torch.nn.functional as F
import torch
from torchvision import transforms



import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class Potsdam(Dataset):
    def __init__(self, args, is_train=True, is_eval=False, res=400):
        """
        res -- resolution of the img, an int in list [400, 800, 1200]
        is_train -- True for training, False for validation and test
        is_eval -- True for test, False for validation
        """
        super().__init__()
        self.args = args
        self.is_train = is_train
        self.is_eval = is_eval
        self.crop_size = int(0.8*res)
        if is_train: # for train or val
            if res == 400:
                self.img_list, self.label_list = self.get_data_list(400)
            elif res == 800:
                self.img_list, self.label_list = self.get_data_list(800)
            elif res == 1200:
                self.img_list, self.label_list = self.get_data_list(1200)       
            else:
                print("wrong resolution in potsdam dataset!") 
        else: # for test or val
            self.img_list, self.label_list = self.get_data_list(0)

        # not using transform, for it can't be applied in both data and label at the same time
        # self.transforms = transforms.Compose([transforms.RandomCrop([256,256]),
        #                                       transforms.RandomHorizontalFlip(p=0.5),
        #                                       transforms.RandomVerticalFlip(p=0.5),])
    
    def get_data_list(self, img_size):
        """
        Description:
            get img list according to the img_size
        Params:
            img_size -- img_size=0 for test, otherwise for train/val
        """
        if img_size != 0:
            # for getting data list
            img_path_temp = os.path.join(self.args.img_dir, 'train_'+ str(img_size) + '.txt')
            label_path_temp = os.path.join(self.args.label_dir, 'train_'+ str(img_size) + '.txt')
            # for loading data
            img_dir_temp = os.path.join(self.args.img_dir, 'train', str(img_size))
            label_dir_temp = os.path.join(self.args.label_dir, 'train', str(img_size))
        else:
            if not self.is_eval:
                img_path_temp = os.path.join(self.args.img_dir, 'val.txt')
                label_path_temp = os.path.join(self.args.label_dir, 'val.txt')
                img_dir_temp = os.path.join(self.args.img_dir, 'val')
                label_dir_temp = os.path.join(self.args.label_dir, 'val')
            else:
                img_path_temp = os.path.join(self.args.img_dir, 'test.txt')
                label_path_temp = os.path.join(self.args.label_dir, 'test.txt')
                img_dir_temp = os.path.join(self.args.img_dir, 'test')
                label_dir_temp = os.path.join(self.args.label_dir, 'test')
        # get the data list -- !!! don't forget to get rid of '\n' in every line of .txt file
        img_list = [os.path.join(img_dir_temp,path.replace('\n','')) for path in open(img_path_temp).readlines()]
        label_list = [os.path.join(label_dir_temp,path.replace('\n','')) for path in open(label_path_temp).readlines()]

        return img_list, label_list

    def rand_crop(self, data, label, img_shape=(256,256)):
        """
        Description:
            crop function
        Params:
            data -- tensor of shape (c,h,w)
            label -- tensor of shape (h,w)
        Tips:
            data should be transformed to shape (c,h,w) firstly !!!
        """
        top = torch.randint(0, data.shape[1] - img_shape[0], size=(1,))[0]
        left = torch.randint(0, data.shape[2] - img_shape[1], size=(1,))[0]
        
                
        data=transforms.functional.crop(data, top, left, img_shape[0], img_shape[1])
        label=transforms.functional.crop(label, top, left, img_shape[0], img_shape[1])
    
        return data, label

    def preprocess(self, img, label, img_shape=(256,256)):
        # To Tensor
        img = torch.from_numpy(img).float()
        label = torch.from_numpy(label).long()

        # cv: h, w, c, tensor: c, h, w
        img = img.permute((2, 0, 1))

        if self.is_train and not self.is_eval:
            img = transforms.functional.center_crop(img, [self.crop_size, self.crop_size])
            label = transforms.functional.center_crop(label, [self.crop_size, self.crop_size])
            ### data augmentation ###
            # crop
            # img, label = self.rand_crop(img, label, img_shape=(self.crop_size,self.crop_size))
            # # flip
            # if torch.rand(1) < 0.5: # horizotal flip
            #     img = transforms.functional.hflip(img)
            #     label = transforms.functional.hflip(label)
            # if torch.rand(1) < 0.5: # vertical flip
            #     img = transforms.functional.vflip(img)
            #     label = transforms.functional.vflip(label)
            pass



        # # label.png to one-hot -- no need to be one_hot for cross entropy loss
        # one_hot_label = F.one_hot(label, num_classes=6) # there are 6 classes in potsdam
        # you can add other processing method or augmentation here
        
        
        return img, label

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_list[idx])[:,:,[2,1,0]] # to rgb
        label = cv2.imread(self.label_list[idx])[:,:,0]

        # preprocessing
        img, label = self.preprocess(img, label)
        return img, label

if __name__ == '__main__':
    ### for test, move the options.py into ./data directory ###
    from options import parse_common_args
    import argparse
    parser = argparse.ArgumentParser()
    parser = parse_common_args(parser)
    args = parser.parse_args()

    dataset = Potsdam(args)
    img, label = dataset[0]
    # for testing
    # label_y = cv2.imread(r'D:\ISPRS\ISPRS 2D Semantic Labeling Contest\potsdam\label_split\train\top_potsdam_2_10_label_0.png')
    x = ''
    # import matplotlib.pyplot as plt
    # plt.imshow(img)
    # plt.xticks([]),plt.yticks([]) # 不显示坐标轴
    # plt.show()
