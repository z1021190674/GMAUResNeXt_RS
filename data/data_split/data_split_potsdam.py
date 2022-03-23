"""
Split the dataset according to the id of the tran/val/test split
"""
import sys
print(sys.path)

from osgeo import gdal_array
from data_split_utils import get_path_list, get_data_split

if __name__ == '__main__':
    ### Potsdam ###
    # the id of the split datasets
    train_id = '2_10, 3_10, 3_11, 3_12, 4_11, 4_12, 5_10, 5_12, 6_8, 6_9, 6_10, 6_11, 6_12, 7_7, 7_9, 7_11, 7_12'
    val_id = '2_11, 2_12, 4_10, 5_11, 6_7, 7_8, 7_10'
    # train_id = '2_10, 3_10, 3_11, 3_12, 4_11, 4_12, 5_10, 5_12, 6_8, 6_9, 6_10, 6_11, 6_12, 7_7, 7_9, 7_11, 7_12, 2_11, 2_12, 4_10, 5_11, 6_7, 7_8, 7_10'
    # val_id = ''
    train_id = train_id.split(', ')
    val_id = val_id.split(', ')

    ### get data split ###
    # get the data list
    dic_data = get_path_list(path = r'D:\ISPRS\ISPRS 2D Semantic Labeling Contest\potsdam\4_Ortho_RGBIR',
                        prefix=r'top_potsdam_',
                        suffix=r'_RGBIR.tif',
                        train_id = train_id,
                        val_id = val_id,)
    train_list = dic_data['train_list']
    test_list = dic_data['test_list']
    val_list = dic_data['val_list']

    get_data_split(train_list, r'D:\ISPRS\ISPRS 2D Semantic Labeling Contest\potsdam\data_split\1200\train',
                   size_samp=(1200, 1200), overlap=600)
    # get_data_split(test_list, r'D:\ISPRS\ISPRS 2D Semantic Labeling Contest\potsdam\data_split\test',
    #                size_samp=(1920, 1920), overlap=960)
    get_data_split(val_list, r'D:\ISPRS\ISPRS 2D Semantic Labeling Contest\potsdam\data_split\1200\val',
                   size_samp=(1200, 1200), overlap=600)

    # ### get colored_label split ###
    # get the colored_label list
    dic_label = get_path_list(path = r'D:\ISPRS\ISPRS 2D Semantic Labeling Contest\potsdam\5_Labels_all',
                        prefix=r'top_potsdam_',
                        suffix=r'_label.tif',
                        train_id = train_id,
                        val_id = val_id,)
    train_label_list = dic_label['train_list']
    test_label_list = dic_label['test_list']
    val_label_list = dic_label['val_list']
    # split the colored label image
    get_data_split(train_label_list, r'D:\ISPRS\ISPRS 2D Semantic Labeling Contest\potsdam\colored_label_split\1200\train',
                   size_samp=(1200, 1200), overlap=600)
    # get_data_split(test_label_list, r'D:\ISPRS\ISPRS 2D Semantic Labeling Contest\potsdam\colored_label_split\test',
    #                size_samp=(1920, 1920), overlap=960)
    get_data_split(val_label_list, r'D:\ISPRS\ISPRS 2D Semantic Labeling Contest\potsdam\colored_label_split\1200\val',
                   size_samp=(1200, 1200), overlap=600)



    # 得到的值是uint8 -- 以numpy array的形式保存？？



    x= ''
    # 读取数据集的路径列表
    # 




    ### 