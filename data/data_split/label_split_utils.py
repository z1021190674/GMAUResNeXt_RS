import numpy as np
import cv2
import os
# from osgeo import gdal_array


def color2label(img, color2label_dict):
    """
    Description:
        transform an colored label image to label image
    Params:
        img -- image of shape (height, width, channel)
        color2label_dict -- a dict of the following type
                            color2label_dict = {
                                "255,255,255": 0,
                                "0,0,255": 1,
                                "0,255,255": 2,
                                "0,255,0": 3,
                                "255,255,0": 4,
                                "255,0,0": 5,
                            }
    Return:
        label -- label of the image according to the color2label_dict,
                 a nparray of shape(height, width)
    """
    label = np.zeros((img.shape[0],img.shape[1]))
    # iterate over the image
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # turn one pixel into label
            pixel_str = str(img[i,j][0]) + ',' + str(img[i,j][1]) + ',' + str(img[i,j][2])
            label[i,j] = color2label_dict[pixel_str]
        print(i)
    return label

def save_color2label(data_list, root_path, target_dir, color2label_dict):
    """
    Description:
        transform the image to label according to the dict, and save it to the target directory
    Params:
        data_list -- data list of the split data
        root_path -- root of the data_list
        target_dir -- the directory to save the label image
        color2label_dict -- the dict of the colored segmentation map to the label
    Notice：
        the data in the data list
    """
    if not os.path.exists(target_dir):
        # os.system('mkdir -p ' + root_path)
        os.makedirs(target_dir)
    for i in range(len(data_list)):
        # get the path
        path = os.path.join(root_path, data_list[i])
        target_path = os.path.join(target_dir, data_list[i])
        # transform color to label
        img = cv2.imread(path)[:,:,[2,1,0]]
        # img = gdal_array.LoadFile(path).transpose(1,2,0)

        label = color2label(img, color2label_dict)

        # for test
        # img =label2color(label.astype(np.uint8))
        # cv2.imshow("img",img)
        # cv2.waitKey()

        cv2.imwrite(target_path, label)


def label2color(label, label2color_dict={}):
    """
    Description:
        transform a label image to a colored label rgb image
    Params:
        label -- label of shape (height, width)
        label2color_dict -- a dict of the following type
                            {
                                "0": [255,255,255],
                                "1": [0,0,255],
                                "2": [0,255,255],
                                "3": [0,255,0],
                                "4": [255,255,0],
                                "5": [255,0,0],
                            }
    Return:
        image -- image of the label according to the label2color_dict,
                 a nparray of shape(height, width, 3)
    """
    if len(label2color_dict) == 0:
        # Potsdam
        label2color_dict = {
                                "0": [255,255,255],
                                "1": [0,0,255],
                                "2": [0,255,255],
                                "3": [0,255,0],
                                "4": [255,255,0],
                                "5": [255,0,0],
                            }
    img = np.zeros((label.shape[0],label.shape[1], 3), dtype=np.uint8)
    # iterate over the image
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            # turn one pixel into rgb image
            pixel_str = str(label[i,j])
            img[i,j] = np.array(label2color_dict[pixel_str])
    return img



if __name__ == '__main__':
    # the default color2label_dict is from Vaihingen and Potsdam datasets
    # 1. Impervious surfaces (RGB: 255, 255, 255)
    # 2. Building (RGB: 0, 0, 255)
    # 3. Low vegetation (RGB: 0, 255, 255)
    # 4. Tree (RGB: 0, 255, 0)
    # 5. Car (RGB: 255, 255, 0)
    # 6. Clutter/background (RGB: 255, 0, 0)
    color2label_dict = {
            "255,255,255": 0,
            "0,0,255": 1,
            "0,255,255": 2,
            "0,255,0": 3,
            "255,255,0": 4,
            "255,0,0": 5,
        }
    x =color2label_dict["255,255,255"]
    # from osgeo import gdal_array
    # img = gdal_array.LoadFile(r'D:\ISPRS\ISPRS 2D Semantic Labeling Contest\potsdam\5_Labels_all\top_potsdam_2_10_label.tif').transpose(1,2,0)
    # label = color2label(img, color2label_dict)
    # y = []