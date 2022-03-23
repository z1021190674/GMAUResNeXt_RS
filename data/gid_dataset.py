"""
Gid dataset
"""



from torch.utils.data import Dataset
import numpy as np
import cv2
import torch.nn.functional as F
import torch
from torchvision import transforms



import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class Gid(Dataset):
    """
        the train/val/test of gid data were all saved in one directory 
    """
    def __init__(self, args, is_train=True, is_eval=False, res=600):
        """
        res -- resolution of the img, an int of 800
        is_train -- True for training, False for validation and test
        is_eval -- True for test, False Ffor validation
        """
        super().__init__()
        self.args = args
        self.is_train = is_train
        self.is_eval = is_eval
        self.crop_size = int(0.8*res)
        
        self.img_list, self.label_list = self.get_data_list(res=res)

        # not using transform, for it can't be applied in both data and label at the same time
        # self.transforms = transforms.Compose([transforms.RandomCrop([256,256]),
        #                                       transforms.RandomHorizontalFlip(p=0.5),
        #                                       transforms.RandomVerticalFlip(p=0.5),])
    
    def get_data_list(self, res=600):
        """logs/Potsdam/AttUResNeXt_class_v2/00/events.out.tfevents.1639050408.ceo-Super-Server.1380415.0
        Description:
            get img list according to the img_size
        Params:
            img_size -- img_size=0 for test, otherwise for train/val
        """
        # for loading data
        img_dir_temp = os.path.join(self.args.img_dir, str(res))
        label_dir_temp = os.path.join(self.args.label_dir, str(res))
        # for getting data list
        if self.is_eval:
            img_path_temp = os.path.join(self.args.img_dir, 'test_' + str(res) + '.txt')
            label_path_temp = os.path.join(self.args.label_dir, 'test_' + str(res) + '.txt')
        else:
            if self.is_train:
                img_path_temp = os.path.join(self.args.img_dir, 'train_' + str(res) + '.txt')
                label_path_temp = os.path.join(self.args.label_dir, 'train_' + str(res) + '.txt')
            else:
                img_path_temp = os.path.join(self.args.img_dir, 'val_' + str(res) + '.txt')
                label_path_temp = os.path.join(self.args.label_dir, 'val_' + str(res) + '.txt')
        # get the data list -- !!! don't forget to get rid of '\n' in every line of .txt file
        img_list = [os.path.join(img_dir_temp,path.replace('\n','')) for path in open(img_path_temp).readlines()]
        label_list = [os.path.join(label_dir_temp,path.replace('\n','')) for path in open(label_path_temp).readlines()]

        return img_list, label_list

    def rand_crop(self, data, label, img_shape=(256,256)):
        """
        Description:
            random crop function
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

    def crop(self, data, label, img_shape=(256,256)):
        """
        Description:
            random crop function
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

    def preprocess(self, img, label):
        # To Tensor
        img = torch.from_numpy(img).float()
        label = torch.from_numpy(label).long()

        # cv: h, w, c, tensor: c, h, w
        img = img.permute((2, 0, 1))

        if self.is_train and not self.is_eval:
            # ### data augmentation ###
            # # crop
            # img, label = self.rand_crop(img, label, img_shape=(self.crop_size,self.crop_size))
            # # flip
            # if torch.rand(1) < 0.5: # horizotal flip
            #     img = transforms.functional.hflip(img)
            #     label = transforms.functional.hflip(label)
            # if torch.rand(1) < 0.5: # vertical flip
            #     img = transforms.functional.vflip(img)
            #     label = transforms.functional.vflip(label)
            pass
        #     # crop
        #     img = transforms.functional.center_crop(img, [self.crop_size, self.crop_size])
        #     label = transforms.functional.center_crop(label, [self.crop_size, self.crop_size])

        if not self.args.is_gpu:
            img.cpu()
            label.cpu()



        # # label.png to one-hot -- no need to be one_hot for cross entropy loss
        # one_hot_label = F.one_hot(label, num_classes=6) # there are 6 classes in Gid
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
    args.img_dir = "/home/zj/data/gid/Fine land-cover Classification_15classes/data_split/" 
    args.label_dir = "/home/zj/data/gid/Fine land-cover Classification_15classes/label_split/" 

    dataset = Gid(args)
    img, label = dataset[0]
    # for testing
    # label_y = cv2.imread(r'D:\ISPRS\ISPRS 2D Semantic Labeling Contest\Gid\label_split\train\top_Gid_2_10_label_0.png')
    x = ''
    # import matplotlib.pyplot as plt
    # plt.imshow(img)
    # plt.xticks([]),plt.yticks([]) # 不显示坐标轴
    # plt.show()
