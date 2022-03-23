#!/bin/bash

# python train.py --model AttUResNeXt_feat_v2 \
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

python train.py --model AttUResNeXt_feat_v2 \
                --exp_id 10 \
                --lr 5e-4 \
                --lr_scheduler ReduceLROnPlateau \
                --dataset Gid \
                --log_dir "./logs/gid_noverlap" \
                --model_dir "checkpoints/gid_noverlap" \
                --num_classes 16 \
                --img_dir "/home/zj/data/gid/Fine land-cover Classification_15classes/data_split/" \
                --label_dir "/home/zj/data/gid/Fine land-cover Classification_15classes/label_split/" \
                --batch_size 4 \
                --epochs 120 \
                --load_model_path "" \
                --res 480 \
                --exp_information "train with res 480"

python train.py --model AttUResNeXt_feat_v2 \
                --exp_id 11 \
                --lr 5e-4 \
                --lr_scheduler ReduceLROnPlateau \
                --dataset Gid \
                --log_dir "./logs/gid_noverlap" \
                --model_dir "checkpoints/gid_noverlap" \
                --num_classes 16 \
                --img_dir "/home/zj/data/gid/Fine land-cover Classification_15classes/data_split/" \
                --label_dir "/home/zj/data/gid/Fine land-cover Classification_15classes/label_split/" \
                --batch_size 4 \
                --epochs 120 \
                --load_model_path "" \
                --res 640 \
                --exp_information "train with res 640"

# # lr schedular -- ReduceLROnPlateau
# for lr in 1e-5 5e-5 1e-4
# do
#     python train.py --model AttUResNeXt_feat_v1 \
#                     --exp_id $exp_id \
#                     --lr $lr \
#                     --lr_scheduler ReduceLROnPlateau \
#                     --dataset Gid \
#                     --log_dir "./logs/gid" \
#                     --model_dir "checkpoints/gid" \
#                     --num_classes 16 \
#                     --img_dir "/home/zj/data/gid/Fine land-cover Classification_15classes/data_split/" \
#                     --label_dir "/home/zj/data/gid/Fine land-cover Classification_15classes/label_split/" \
#                     --batch_size 4 \
#                     --exp_information "train with res 400 600 800"
#     exp_id=$(($exp_id+1))
# done

# exp_id=0
# # lr schedular -- ReduceLROnPlateau
# for lr in 1e-5 5e-5 1e-4
# do
#     python train.py --model AttUResNeXt_feat_v2 \
#                     --exp_id $exp_id \
#                     --lr $lr \
#                     --lr_scheduler ReduceLROnPlateau \
#                     --dataset Gid \
#                     --log_dir "./logs/gid" \
#                     --model_dir "checkpoints/gid" \
#                     --num_classes 16 \
#                     --img_dir "/home/zj/data/gid/Fine land-cover Classification_15classes/data_split/" \
#                     --label_dir "/home/zj/data/gid/Fine land-cover Classification_15classes/label_split/" \
#                     --batch_size 4 \
#                     --load_model_path "" \
#                     --res 800 \
#                     --exp_information "train with res 800"
#     exp_id=$(($exp_id+1))
# done
