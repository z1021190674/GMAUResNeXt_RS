
from osgeo import gdal
import numpy as np
np.set_printoptions(threshold=np.inf) #使print大量数据不用符号...代替而显示所有

dataset = gdal.Open(r"D:\ISPRS\ISPRS 2D Semantic Labeling Contest\potsdam\4_Ortho_RGBIR\top_potsdam_2_10_RGBIR.tif")

# from osgeo import gdal_array
# rasterArray = gdal_array.LoadFile(r"D:\ISPRS\ISPRS 2D Semantic Labeling Contest\potsdam\4_Ortho_RGBIR\top_potsdam_2_10_RGBIR.tif")

print(dataset.GetDescription())#数据描述

print(dataset.RasterCount)#波段数

cols=dataset.RasterXSize#图像长度
rows=(dataset.RasterYSize)#图像宽度

xoffset=cols/2
yoffset=rows/2

band = dataset.GetRasterBand(3)#取第三波段
r=band.ReadAsArray(xoffset,yoffset,1000,1000)#从数据的中心位置位置开始，取1000行1000列数据

band = dataset.GetRasterBand(2)
g=band.ReadAsArray(xoffset,yoffset,1000,1000)

band = dataset.GetRasterBand(1)
b=band.ReadAsArray(xoffset,yoffset,1000,1000)

import cv2
import matplotlib.pyplot as plt

type(b)
img2=cv2.merge([r,g,b])
plt.imshow(img2)
plt.xticks([]),plt.yticks([]) # 不显示坐标轴
plt.show()
