#!/bin/bash
# screen -S eval_cnn
# python eval.py --model AttUResNeXt_class_v2 \
#                --load_model_path  "checkpoints/gid_noverlap/AttUResNeXt_class_v2/12/best/AttUResNeXt_class_v2_088.pth.tar" \
#                --gpu_id 1 \
#                --img_dir "/home/zj/data/gid/Fine land-cover Classification_15classes/data_split/" \
#                --label_dir "/home/zj/data/gid/Fine land-cover Classification_15classes/label_split/" \
#                --res 320 \
#                --num_classes 16 \
#                --dataset Gid \
#                --batch_size 8 \
#                --is_tta True

# python eval.py --model AttUResNeXt_class_v2 \
#                --load_model_path  "checkpoints/postdam/AttUResNeXt_class_v2/09/best/AttUResNeXt_class_v2_085.pth.tar" \
#                --gpu_id 0 \
#                --img_dir "/home/zj/data/potsdam/data_split/" \
#                --label_dir "/home/zj/data/potsdam/label_split/" \
#                --num_classes 6 \
#                --dataset Potsdam \
#                --batch_size 4 \
#                --is_tta True

    # parser.add_argument('--img_dir', type=str,
    #                     default="/home/zj/data/potsdam/data_split/",
    #                     help='the directory path of the dataset, the .txt file is in the corresponding root directory!!!')
    # parser.add_argument('--label_dir', type=str,
    #                     default="/home/zj/data/potsdam/label_split/",
    #                     help='the directory path of the labels,  the .txt file is in the corresponding root directory!!!')

# python eval.py --model AttUResNeXt_class_v2 \
#                --load_model_path  "checkpoints/gid_noverlap/AttUResNeXt_class_v2/12/best/AttUResNeXt_class_v2_088.pth.tar" \
#                --gpu_id 0 \
#                --img_dir "/home/zj/data/gid/Fine land-cover Classification_15classes/data_split/" \
#                --label_dir "/home/zj/data/gid/Fine land-cover Classification_15classes/label_split/" \
#                --res 320 \
#                --num_classes 16 \
#                --dataset Gid \
#                --batch_size 8 \
#                --is_tta False

# python eval.py --model PSPNet \
#                --load_model_path  "checkpoints/gid_noverlap/PSPNet/12/best/PSPNet_002.pth.tar" \
#                --gpu_id 0 \
#                --img_dir "/home/zj/data/gid/Fine land-cover Classification_15classes/data_split/" \
#                --label_dir "/home/zj/data/gid/Fine land-cover Classification_15classes/label_split/" \
#                --res 320 \
#                --num_classes 16 \
#                --dataset Gid \
#                --batch_size 8 \
#                --is_tta False

# python eval.py --model NestedUNet \
#                --load_model_path  "/home/zj/scene_seg/checkpoints/gid_noverlap/NestedUNet/12/best/NestedUNet_012.pth.tar" \
#                --gpu_id 0 \
#                --img_dir "/home/zj/data/gid/Fine land-cover Classification_15classes/data_split/" \
#                --label_dir "/home/zj/data/gid/Fine land-cover Classification_15classes/label_split/" \
#                --res 320 \
#                --num_classes 16 \
#                --dataset Gid \
#                --batch_size 8 \
#                --is_tta False

# python eval.py --model FastFCN \
#                --load_model_path  "checkpoints/gid_noverlap/FastFCN/12/best/FastFCN_005.pth.tar" \
#                --gpu_id 0 \
#                --img_dir "/home/zj/data/gid/Fine land-cover Classification_15classes/data_split/" \
#                --label_dir "/home/zj/data/gid/Fine land-cover Classification_15classes/label_split/" \
#                --res 320 \
#                --num_classes 16 \
#                --dataset Gid \
#                --batch_size 8 \
#                --is_tta False

# python eval.py --model DeepLabV3plus \
#                --load_model_path  "checkpoints/gid_noverlap/DeepLabV3plus/12/best/DeepLabV3plus_005.pth.tar" \
#                --gpu_id 0 \
#                --img_dir "/home/zj/data/gid/Fine land-cover Classification_15classes/data_split/" \
#                --label_dir "/home/zj/data/gid/Fine land-cover Classification_15classes/label_split/" \
#                --res 320 \
#                --num_classes 16 \
#                --dataset Gid \
#                --batch_size 8 \
#                --is_tta False

python eval.py --model UResNeXt \
               --load_model_path  "checkpoints/gid_noverlap/UResNeXt/12/best/UResNeXt_086.pth.tar" \
               --gpu_id 0 \
               --img_dir "/home/zj/data/gid/Fine land-cover Classification_15classes/data_split/" \
               --label_dir "/home/zj/data/gid/Fine land-cover Classification_15classes/label_split/" \
               --res 320 \
               --num_classes 16 \
               --dataset Gid \
               --batch_size 8 \
               --is_tta False