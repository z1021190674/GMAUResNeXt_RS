#!/bin/bash

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

exp_id=0
# lr schedular -- ReduceLROnPlateau
for lr in 1e-5 5e-5 1e-4
do
    python train.py --model AttUResNeXt_feat_v2 \
                    --exp_id $exp_id \
                    --lr $lr \
                    --lr_scheduler ReduceLROnPlateau
    exp_id=$(($exp_id+1))
done

# exp_id=0
# # lr schedular -- ReduceLROnPlateau
# for lr in 1e-5 5e-5 1e-4
# do
#     python train.py --model AttUResNeXt_feat_v3 \
#                     --exp_id $exp_id \
#                     --lr $lr \
#                     --lr_scheduler ReduceLROnPlateau
#     exp_id=$(($exp_id+1))
# done