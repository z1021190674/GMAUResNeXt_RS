#!/bin/bash

# 

# python train.py --model AttUResNeXt_class_v2 \
#                 --exp_id 9 \
#                 --lr 5e-4 \
#                 --lr_scheduler ReduceLROnPlateau \
#                 --dataset Gid \
#                 --log_dir "./logs/gid_noverlap" \
#                 --model_dir "checkpoints/gid_noverlap" \
#                 --num_classes 16 \
#                 --img_dir "/home/zj/data/gid/Fine land-cover Classification_15classes/data_split/" \
#                 --label_dir "/home/zj/data/gid/Fine land-cover Classification_15classes/label_split/" \
#                 --batch_size 4 \
#                 --epochs 90 \
#                 --load_model_path "" \
#                 --res 320 \
#                 --exp_information "train with res 320"

# python train.py --model AttUResNeXt_class_v2 \
#                 --exp_id 10 \
#                 --lr 5e-4 \
#                 --lr_scheduler ReduceLROnPlateau \
#                 --dataset Gid \
#                 --log_dir "./logs/gid_noverlap" \
#                 --model_dir "checkpoints/gid_noverlap" \
#                 --num_classes 16 \
#                 --img_dir "/home/zj/data/gid/Fine land-cover Classification_15classes/data_split/" \
#                 --label_dir "/home/zj/data/gid/Fine land-cover Classification_15classes/label_split/" \
#                 --batch_size 4 \
#                 --epochs 120 \
#                 --load_model_path "" \
#                 --res 480 \
#                 --exp_information "train with res 480"

# python train.py --model AttUResNeXt_class_v2 \
#                 --exp_id 11 \
#                 --lr 5e-4 \
#                 --lr_scheduler ReduceLROnPlateau \
#                 --dataset Gid \
#                 --log_dir "./logs/gid_noverlap" \
#                 --model_dir "checkpoints/gid_noverlap" \
#                 --num_classes 16 \
#                 --img_dir "/home/zj/data/gid/Fine land-cover Classification_15classes/data_split/" \
#                 --label_dir "/home/zj/data/gid/Fine land-cover Classification_15classes/label_split/" \
#                 --batch_size 4 \
#                 --epochs 120 \
#                 --load_model_path "" \
#                 --res 640 \
#                 --exp_information "train with res 640"

python train.py --model R2U_Net \
                --exp_id 12 \
                --lr 1e-4 \
                --lr_scheduler ReduceLROnPlateau \
                --dataset Gid \
                --log_dir "./logs/gid_noverlap" \
                --model_dir "checkpoints/gid_noverlap" \
                --num_classes 16 \
                --img_dir "/home/zj/data/gid/Fine land-cover Classification_15classes/data_split/" \
                --label_dir "/home/zj/data/gid/Fine land-cover Classification_15classes/label_split/" \
                --batch_size 2 \
                --epochs 90 \
                --load_model_path "" \
                --res 320 \
                --exp_information "train with res 320"

python train.py --model R2AttU_Net \
                --exp_id 12 \
                --lr 1e-4 \
                --lr_scheduler ReduceLROnPlateau \
                --dataset Gid \
                --log_dir "./logs/gid_noverlap" \
                --model_dir "checkpoints/gid_noverlap" \
                --num_classes 16 \
                --img_dir "/home/zj/data/gid/Fine land-cover Classification_15classes/data_split/" \
                --label_dir "/home/zj/data/gid/Fine land-cover Classification_15classes/label_split/" \
                --batch_size 2 \
                --epochs 90 \
                --load_model_path "" \
                --res 320 \
                --exp_information "train with res 320"

python train.py --model UResNeXt \
                --exp_id 12 \
                --lr 1e-4 \
                --lr_scheduler ReduceLROnPlateau \
                --dataset Gid \
                --log_dir "./logs/gid_noverlap" \
                --model_dir "checkpoints/gid_noverlap" \
                --num_classes 16 \
                --img_dir "/home/zj/data/gid/Fine land-cover Classification_15classes/data_split/" \
                --label_dir "/home/zj/data/gid/Fine land-cover Classification_15classes/label_split/" \
                --batch_size 4 \
                --epochs 90 \
                --load_model_path "" \
                --res 320 \
                --exp_information "train with res 320"

# python train.py --model NestedUNet \
#                 --exp_id 12 \
#                 --lr 1e-4 \
#                 --lr_scheduler ReduceLROnPlateau \
#                 --dataset Gid \
#                 --log_dir "./logs/gid_noverlap" \
#                 --model_dir "checkpoints/gid_noverlap" \
#                 --num_classes 16 \
#                 --img_dir "/home/zj/data/gid/Fine land-cover Classification_15classes/data_split/" \
#                 --label_dir "/home/zj/data/gid/Fine land-cover Classification_15classes/label_split/" \
#                 --batch_size 4 \
#                 --epochs 90 \
#                 --load_model_path "" \
#                 --res 320 \
#                 --exp_information "train with res 320"

# python train.py --model AttU_Net \
#                 --exp_id 12 \
#                 --lr 1e-4 \
#                 --lr_scheduler ReduceLROnPlateau \
#                 --dataset Gid \
#                 --log_dir "./logs/gid_noverlap" \
#                 --model_dir "checkpoints/gid_noverlap" \
#                 --num_classes 16 \
#                 --img_dir "/home/zj/data/gid/Fine land-cover Classification_15classes/data_split/" \
#                 --label_dir "/home/zj/data/gid/Fine land-cover Classification_15classes/label_split/" \
#                 --batch_size 4 \
#                 --epochs 90 \
#                 --load_model_path "" \
#                 --res 320 \
#                 --exp_information "train with res 320"

# python train.py --model PSPNet \
#                 --exp_id 12 \
#                 --lr 1e-4 \
#                 --lr_scheduler ReduceLROnPlateau \
#                 --dataset Gid \
#                 --log_dir "./logs/gid_noverlap" \
#                 --model_dir "checkpoints/gid_noverlap" \
#                 --num_classes 16 \
#                 --img_dir "/home/zj/data/gid/Fine land-cover Classification_15classes/data_split/" \
#                 --label_dir "/home/zj/data/gid/Fine land-cover Classification_15classes/label_split/" \
#                 --batch_size 4 \
#                 --epochs 90 \
#                 --load_model_path "" \
#                 --res 320 \
#                 --exp_information "train with res 320"

# python train.py --model FastFCN \
#                 --exp_id 12 \
#                 --lr 1e-4 \
#                 --lr_scheduler ReduceLROnPlateau \
#                 --dataset Gid \
#                 --log_dir "./logs/gid_noverlap" \
#                 --model_dir "checkpoints/gid_noverlap" \
#                 --num_classes 16 \
#                 --img_dir "/home/zj/data/gid/Fine land-cover Classification_15classes/data_split/" \
#                 --label_dir "/home/zj/data/gid/Fine land-cover Classification_15classes/label_split/" \
#                 --batch_size 4 \
#                 --epochs 90 \
#                 --load_model_path "" \
#                 --res 320 \
#                 --exp_information "train with res 320"

# python train.py --model DeepLabV3plus \
#                 --exp_id 12 \
#                 --lr 1e-4 \
#                 --lr_scheduler ReduceLROnPlateau \
#                 --dataset Gid \
#                 --log_dir "./logs/gid_noverlap" \
#                 --model_dir "checkpoints/gid_noverlap" \
#                 --num_classes 16 \
#                 --img_dir "/home/zj/data/gid/Fine land-cover Classification_15classes/data_split/" \
#                 --label_dir "/home/zj/data/gid/Fine land-cover Classification_15classes/label_split/" \
#                 --batch_size 4 \
#                 --epochs 90 \
#                 --load_model_path "" \
#                 --res 320 \
#                 --exp_information "train with res 320"

# python train.py --model DANet \
#                 --exp_id 12 \
#                 --lr 1e-4 \
#                 --lr_scheduler ReduceLROnPlateau \
#                 --dataset Gid \
#                 --log_dir "./logs/gid_noverlap" \
#                 --model_dir "checkpoints/gid_noverlap" \
#                 --num_classes 16 \
#                 --img_dir "/home/zj/data/gid/Fine land-cover Classification_15classes/data_split/" \
#                 --label_dir "/home/zj/data/gid/Fine land-cover Classification_15classes/label_split/" \
#                 --batch_size 4 \
#                 --epochs 90 \
#                 --load_model_path "" \
#                 --res 320 \
#                 --exp_information "train with res 320"

# python train.py --model AttUResNeXt_class_v2 \
#                 --exp_id 13 \
#                 --lr 1e-4 \
#                 --lr_scheduler ReduceLROnPlateau \
#                 --dataset Gid \
#                 --log_dir "./logs/gid_noverlap" \
#                 --model_dir "checkpoints/gid_noverlap" \
#                 --num_classes 16 \
#                 --img_dir "/home/zj/data/gid/Fine land-cover Classification_15classes/data_split/" \
#                 --label_dir "/home/zj/data/gid/Fine land-cover Classification_15classes/label_split/" \
#                 --batch_size 4 \
#                 --epochs 120 \
#                 --load_model_path "" \
#                 --res 480 \
#                 --exp_information "train with res 480"

# python train.py --model AttUResNeXt_class_v2 \
#                 --exp_id 14 \
#                 --lr 1e-4 \
#                 --lr_scheduler ReduceLROnPlateau \
#                 --dataset Gid \
#                 --log_dir "./logs/gid_noverlap" \
#                 --model_dir "checkpoints/gid_noverlap" \
#                 --num_classes 16 \
#                 --img_dir "/home/zj/data/gid/Fine land-cover Classification_15classes/data_split/" \
#                 --label_dir "/home/zj/data/gid/Fine land-cover Classification_15classes/label_split/" \
#                 --batch_size 4 \
#                 --epochs 120 \
#                 --load_model_path "" \
#                 --res 640 \
#                 --exp_information "train with res 640"


# exp_id=6
# # lr schedular -- ReduceLROnPlateau
# for lr in 1e-3 5e-4 1e-4
# do
#     python train.py --model AttUResNeXt_class_v1 \
#                     --exp_id $exp_id \
#                     --lr $lr \
#                     --lr_scheduler ReduceLROnPlateau \
#                     --dataset Gid \
#                     --log_dir "./logs/gid_noverlap" \
#                     --model_dir "checkpoints/gid_noverlap" \
#                     --num_classes 16 \
#                     --img_dir "/home/zj/data/gid/Fine land-cover Classification_15classes/data_split/" \
#                     --label_dir "/home/zj/data/gid/Fine land-cover Classification_15classes/label_split/" \
#                     --batch_size 4 \
#                     --epochs 90 \
#                     --load_model_path "" \
#                     --res 320 \
#                     --exp_information "train with res 320"
#     exp_id=$(($exp_id+1))
# done

# # AttUResNeXt_class_v2, exp_id 0~3 for ReduceLROnPlateau, 4~7 forCosineOneCircle
# exp_id=6
# # lr schedular -- ReduceLROnPlateau
# for lr in 1e-3 5e-4 1e-4
# do
#     python train.py --model AttUResNeXt_class_v2 \
#                     --exp_id $exp_id \
#                     --lr $lr \
#                     --lr_scheduler ReduceLROnPlateau \
#                     --dataset Gid \
#                     --log_dir "./logs/gid_noverlap" \
#                     --model_dir "checkpoints/gid_noverlap" \
#                     --num_classes 16 \
#                     --img_dir "/home/zj/data/gid/Fine land-cover Classification_15classes/data_split/" \
#                     --label_dir "/home/zj/data/gid/Fine land-cover Classification_15classes/label_split/" \
#                     --batch_size 4 \
#                     --epochs 120 \
#                     --load_model_path "" \
#                     --res 320 \
#                     --exp_information "train with res 320"
#     exp_id=$(($exp_id+1))
# done
