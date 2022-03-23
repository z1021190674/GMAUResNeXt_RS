#!/bin/bash

python train.py --model AttUResNeXt_feat_v1 \
                --exp_id 9 \
                --lr 1e-4 \
                --lr_scheduler ReduceLROnPlateau \
                --dataset "Potsdam" \
                --num_classes 6 \
                --img_dir "/home/zj/data/potsdam/data_split/" \
                --label_dir "/home/zj/data/potsdam/label_split/" \
                --model_dir 'checkpoints/postdam' \
                --log_dir 'logs/potsdam' \
                --load_model_path '' \
                --gpu_id '1' \
                --batch_size 2 

python train.py --model AttUResNeXt_feat_v2 \
                --exp_id 9 \
                --lr 1e-4 \
                --lr_scheduler ReduceLROnPlateau \
                --dataset "Potsdam" \
                --num_classes 6 \
                --img_dir "/home/zj/data/potsdam/data_split/" \
                --label_dir "/home/zj/data/potsdam/label_split/" \
                --model_dir 'checkpoints/postdam' \
                --log_dir 'logs/potsdam' \
                --load_model_path '' \
                --gpu_id '1' \
                --batch_size 2 

# exp_id=0
# # lr schedular -- ReduceLROnPlateau
# for lr in 1e-5 5e-5 1e-4
# do
#     python train.py --model AttUResNeXt_feat_v1 \
#                     --exp_id $exp_id \
#                     --lr $lr \
#                     --lr_scheduler ReduceLROnPlateau
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
#                     --dataset "Potsdam" \
#                     --num_classes 6 \
#                     --img_dir "/home/zj/data/potsdam/data_split/" \
#                     --label_dir "/home/zj/data/potsdam/label_split/" \
#                     --model_dir 'checkpoints/postdam'
#     exp_id=$(($exp_id+1))
# done
