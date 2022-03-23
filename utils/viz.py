import cv2
import numpy as np


def label2rgb(label_np):
    """
    Description:
        get the colormap of the predicted label
    Params:
        label_np -- np array of shape (c,h,w), c is the number of classes
    Return:
        label_color -- the colormap, np array of shape (h,w,3)
    """
    # label_np of shape(c,h,w) --> label_color = [[class_0_0, ... , class_0_w-1],
    #                                              class_0_1, ... , class_1_w-1],
    #                                                          .
    #                                                          .
    #                                                          .
    #                                              class_h-1_1, ... , class_h-1_w-1]]
    label_color = np.argmax(label_np, axis=0)
    label_color = label_color / label_np.shape[0] * 255
    label_color = cv2.applyColorMap(label_color.astype(np.uint8), cv2.COLORMAP_JET)
    return label_color

def label2color(label, dataset):
    """
    Description:
        transform a label image to a colored label rgb image
    Params:
        label -- label of shape (height, width)
        label2color_dict(Potsdam) -- a dict of the following type
                            color2label_dict = {
                                "255,255,255": 0,
                                "0,0,255": 1,
                                "0,255,255": 2,
                                "0,255,0": 3,
                                "255,255,0": 4,
                                "255,0,0": 5,
                            }
                            Impervious surfaces (RGB: 255, 255, 255)
                            Building (RGB: 0, 0, 255)
                            Low vegetation (RGB: 0, 255, 255)
                            Tree (RGB: 0, 255, 0)
                            Car (RGB: 255, 255, 0)
                            Clutter/background (RGB: 255, 0, 0)
    Return:
        image -- image of the label according to the label2color_dict,
                 a nparray of shape(height, width, 3)
    """
    if dataset == 'Potsdam':
        # Potsdam
        label2color_dict = {
                                "0": [255,255,255],
                                "1": [0,0,255],
                                "2": [0,255,255],
                                "3": [0,255,0],
                                "4": [255,255,0],
                                "5": [255,0,0],
                            }
    elif dataset == 'Gid':
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
                                "12": [0,0,500],
                                "13": [0,150,200],
                                "14": [0,200,250],
                                "15": [0,0,0],
                            }
    img = np.zeros((label.shape[0],label.shape[1], 3), dtype=np.uint8)
    # iterate over the image
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            # turn one pixel into rgb image
            pixel_str = str(label[i,j])
            img[i,j] = np.array(label2color_dict[pixel_str])
    return img