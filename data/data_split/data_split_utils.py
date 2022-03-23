"""
Implemented in Windows10
get_path_list -- to get the data list according to the data id of the train/val/test split
get_data_split -- generate the split data to the desired directory
"""

import os
import cv2
from osgeo import gdal_array

def get_list(id ,prefix='', suffix=''):
    """
    Description:
        get the data list according to the id of data
    Parameters:
        id -- the id of the acquired data 
        prefix -- prefix of the data path
        suffix -- suffix of the data path
    """
    data_list = [i.join((prefix, suffix)) for i in id]
    return data_list


def get_path_list(path, train_id, val_id, prefix, suffix, is_test = True):
    """
    Description:
        get the path list of the data given the id of training and validating data
    Parameters:
        path -- root path of the data
        train_id -- list of id of the training data, like ['1_1','1_2',...]
        val_id -- list of id of the validating data, like ['1_1','1_2',...]
        test_id -- the default test id is not given, thus the default test_id is of []
        prefix -- prefix of the tif file
        sufix -- suffix of the tif file
    Return:
        a dict of the path list of train/val/test data
    """
    ### get the path list of the data ###
    # get rid of '.tfw' file
    data_list = os.listdir(path)
    temp_list = []
    for i in range(len(data_list)):
        if data_list[i].find('.tif') != -1: # in potsdam there is a .tfw file according to its .tif file
            temp_list.append(data_list[i])
    data_list = temp_list

    ### When not giving test_id ###
    if is_test:
        test_id = []
        # get all data id
        data_id = [i.replace(prefix, '') for i in data_list]
        data_id = [i.replace(suffix, '') for i in data_id]
        # get the id of the test data
        for i in range(len(train_id)):
            data_id.remove(train_id[i])
        for i in range(len(val_id)):
            data_id.remove(val_id[i])
        test_id = data_id
    # get the list of the split data
    train_list = get_list(train_id, prefix=path + '\\' + prefix, suffix=suffix)
    val_list = get_list(val_id, prefix=path + '\\' + prefix, suffix=suffix)
    test_list = get_list(test_id, prefix=path + '\\' + prefix, suffix=suffix)
    return {'train_list':train_list,
            'test_list':test_list,
            'val_list':val_list}


def save_crop(img, root_path, data_name, size_samp = (384,384), overlap = 72, is_data = True):
    """
    Description:
        to save the cropped image given the image
    Params:
        root_path -- the root path to save the cropped image
        data_name -- the primitive name of the data
        size_samp -- the size of the cropped image
        overlap -- the overlap size of the cropped image
        is_data -- 1-data, 0-label
    """
    mark_h = 0
    mark_w = 0
    height = 0
    width = 0
    id = 0
    suffix = '.' + data_name.split('.')[-1]
    if not os.path.exists(root_path):
        # os.system('mkdir -p ' + root_path)
        os.makedirs(root_path)
    # slide to crop
    if is_data:
        while mark_h == 0:
            while mark_w == 0:
                cropped_img = img[:3,height:height+size_samp[0],width:width+size_samp[1]].transpose(1,2,0)
                # import matplotlib.pyplot as plt # for test
                # plt.imshow(cropped_img)
                # plt.xticks([]),plt.yticks([]) # 不显示坐标轴
                # plt.show()
                # save the cropped image -- save the cropped image to the corresponding path
                name = data_name.replace(suffix, '_' + str(id) + '.png')
                path = os.path.join(root_path, name)
                cv2.imwrite(path, cropped_img[:,:,[2,1,0]])
                # next image, sliding from left to right (use mark_w to judge whether it is overstepping the boundary)
                width += +size_samp[1] - overlap
                if width + size_samp[1] >= img.shape[2]:
                    mark_w = 1
                else: 
                    mark_w = 0
                    id += 1
            # sliding down the img (use mark_h to judge whether it is overstepping the boundary
            height += size_samp[0] - overlap
            if height + size_samp[0] >= img.shape[1]:
                    mark_h = 1
            else: 
                # reset the width
                mark_w = 0
                width = 0
                id += 1


def get_data_split(data_list, root_path, size_samp=(384,384), overlap=72):
    """
    Description:
        generate the split data to the desired directory
    Notice:
        it uses gdal_array to load .tif file
    Params:
        datalist -- elements of datalist is  "...\\xxx.tif"
        root_path -- the root path to save the cropped image
        size_samp -- the size of the cropped image
        overlap -- the overlap size of the cropped image
    """
    for i in range(len(data_list)):
        img = gdal_array.LoadFile(data_list[i])
        data_name = data_list[i].split('\\')[-1]
        save_crop(img, root_path, data_name, size_samp, overlap)