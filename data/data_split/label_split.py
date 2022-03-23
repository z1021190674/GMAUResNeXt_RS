"""
transform the split colored label image into label
"""
from label_split_utils import save_color2label
import os

if __name__ == '__main__':
    # the dict of potsdam and vaihingen
    color2label_dict = {
            "255,255,255": 0,
            "0,0,255": 1,
            "0,255,255": 2,
            "0,255,0": 3,
            "255,255,0": 4,
            "252,255,0":4, # there are some [252,255,0] values in the dataset
            "255,0,0": 5,
        }

    # root of the split data
    train_root = "/home/zj/data/potsdam/colored_label_split/400/train"
    # test_root = "/home/zj/data/potsdam/colored_label_split/test"
    val_root = "/home/zj/data/potsdam/colored_label_split/400/val"
    # get the list of the split data
    train_label_list = os.listdir(train_root)
    # test_label_list = os.listdir(test_root)
    val_label_list = os.listdir(val_root)
    # train
    save_color2label(train_label_list,
                     root_path=train_root,
                     target_dir="/home/zj/data/potsdam/label_split/400/train",
                     color2label_dict=color2label_dict)
    # validation
    save_color2label(val_label_list,
                     root_path=val_root,
                     target_dir="/home/zj/data/potsdam/label_split/400/val",
                     color2label_dict=color2label_dict)
    # # test
    # save_color2label(test_label_list,
    #                  root_path=test_root,
    #                  target_dir="/home/zj/data/potsdam/label_split/test",
    #                  color2label_dict=color2label_dict)

