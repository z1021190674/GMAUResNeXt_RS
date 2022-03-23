"""
Split the dataset according to the id of the tran/val/test split
"""

from data_split_utils import get_path_list
from osgeo import gdal_array

### Vaihingen ###
# id of the split
train_id = '1, 3, 5, 7, 13, 17, 21, 23, 26, 32, 37'
val_id = '11, 15, 28, 30, 34'
train_id = train_id.split(', ')
val_id = val_id.split(', ')

# get the path list
dic_data = get_path_list(path = r'D:\ISPRS\ISPRS 2D Semantic Labeling Contest\vaihingen\ISPRS_semantic_labeling_Vaihingen\top',
                    prefix=r'top_mosaic_09cm_area',
                    suffix=r'.tif',
                    train_id = train_id,
                    val_id = val_id,)
train_list = dic_data['train_list']
test_list = dic_data['test_list']
val_list = dic_data['val_list']

# 测试 -- 展示读到的图像 #
# filepath_t = {'1':r'D:\ISPRS\ISPRS 2D Semantic Labeling Contest\vaihingen\ISPRS_semantic_labeling_Vaihingen\top\top_mosaic_09cm_area1.tif',
#             '2':r"D:\ISPRS\ISPRS 2D Semantic Labeling Contest\potsdam\4_Ortho_RGBIR\top_potsdam_2_10_RGBIR.tif",
#             '3':r'D:\ISPRS\ISPRS 2D Semantic Labeling Contest\potsdam\4_Ortho_RGBIR\top_mosaic_09cm_area1.tif'}
rasterArray = gdal_array.LoadFile(train_list[0])
import matplotlib.pyplot as plt
plt.imshow(rasterArray[:3,0:600,0:600].transpose(1,2,0))
plt.xticks([]),plt.yticks([]) # 不显示坐标轴
plt.show()

# 得到的值是uint8 -- 以numpy array的形式保存？？

x= ''