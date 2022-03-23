import os

def write_data_txt(root_dir, txt_path):
    """
    Description:
        write the data filename into one .txt file
    Params:
        root_dir -- root directory of the data
        txt_path -- the path to save the txt file
    """
    data_list = os.listdir(root_dir)
    data_list.sort()
    with open(txt_path, 'w') as f:
        for i in range(len(data_list) - 1): # for using the f.writelines
            data_list[i] = data_list[i] + '\n'
        f.writelines(data_list)
    x= ''

if __name__ == '__main__':
    ### data ###
    # test
    root_dir = "/home/zj/data/potsdam/data_split/test/"
    txt_file = "/home/zj/data/potsdam/data_split/test.txt"
    write_data_txt(root_dir, txt_file)
    # val
    root_dir = "/home/zj/data/potsdam/data_split/val/"
    txt_file = "/home/zj/data/potsdam/data_split/val.txt"
    write_data_txt(root_dir, txt_file)
    # train
    root_dir = "/home/zj/data/potsdam/data_split/train/1200"
    txt_file = "/home/zj/data/potsdam/data_split/train_1200.txt"
    write_data_txt(root_dir, txt_file)
    root_dir = "/home/zj/data/potsdam/data_split/train/800"
    txt_file = "/home/zj/data/potsdam/data_split/train_800.txt"
    write_data_txt(root_dir, txt_file)
    oot_dir = "/home/zj/data/potsdam/data_split/train/400"
    txt_file = "/home/zj/data/potsdam/data_split/train_400.txt"
    write_data_txt(root_dir, txt_file)


    ### label ###
    root_dir = "/home/zj/data/potsdam/label_split/test/"
    txt_file = "/home/zj/data/potsdam/label_split/test.txt"
    write_data_txt(root_dir, txt_file)
    # val
    root_dir = "/home/zj/data/potsdam/label_split/val/"
    txt_file = "/home/zj/data/potsdam/label_split/val.txt"
    write_data_txt(root_dir, txt_file)
    # train
    root_dir = "/home/zj/data/potsdam/label_split/train/1200"
    txt_file = "/home/zj/data/potsdam/label_split/train_1200.txt"
    write_data_txt(root_dir, txt_file)
    root_dir = "/home/zj/data/potsdam/label_split/train/800"
    txt_file = "/home/zj/data/potsdam/label_split/train_800.txt"
    write_data_txt(root_dir, txt_file)
    oot_dir = "/home/zj/data/potsdam/label_split/train/400"
    txt_file = "/home/zj/data/potsdam/label_split/train_400.txt"
    write_data_txt(root_dir, txt_file)