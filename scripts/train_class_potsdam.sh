# python train.py --model AttUResNeXt_class_v1 \
#                 --exp_id 9 \
#                 --lr 1e-4 \
#                 --lr_scheduler ReduceLROnPlateau \
#                 --dataset "Potsdam" \
#                 --num_classes 6 \
#                 --img_dir "/home/zj/data/potsdam/data_split/" \
#                 --label_dir "/home/zj/data/potsdam/label_split/" \
#                 --model_dir 'checkpoints/postdam' \
#                 --log_dir 'logs/potsdam' \
#                 --load_model_path '' \
#                 --gpu_id '1' \
#                 --batch_size 2 \


python train.py --model AttUResNeXt_class_v2 \
                --exp_id 10 \
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


# python train.py --model UResNeXt \
#                 --exp_id 9 \
#                 --lr 1e-4 \
#                 --lr_scheduler ReduceLROnPlateau \
#                 --dataset "Potsdam" \
#                 --num_classes 6 \
#                 --img_dir "/home/zj/data/potsdam/data_split/" \
#                 --label_dir "/home/zj/data/potsdam/label_split/" \
#                 --model_dir 'checkpoints/postdam' \
#                 --log_dir 'logs/potsdam' \
#                 --load_model_path '' \
#                 --gpu_id '1' \
#                 --batch_size 2 


# python train.py --model AttUResNeXt_class_v2_3 \
#                 --exp_id 9 \
#                 --lr 1e-4 \
#                 --lr_scheduler ReduceLROnPlateau \
#                 --dataset "Potsdam" \
#                 --num_classes 6 \
#                 --img_dir "/home/zj/data/potsdam/data_split/" \
#                 --label_dir "/home/zj/data/potsdam/label_split/" \
#                 --model_dir 'checkpoints/postdam' \
#                 --log_dir 'logs/potsdam' \
#                 --load_model_path '' \
#                 --gpu_id '1' \
#                 --batch_size 2 

# python train.py --model AttUResNeXt_class_v2_2 \
#                 --exp_id 9 \
#                 --lr 1e-4 \
#                 --lr_scheduler ReduceLROnPlateau \
#                 --dataset "Potsdam" \
#                 --num_classes 6 \
#                 --img_dir "/home/zj/data/potsdam/data_split/" \
#                 --label_dir "/home/zj/data/potsdam/label_split/" \
#                 --model_dir 'checkpoints/postdam' \
#                 --log_dir 'logs/potsdam' \
#                 --load_model_path '' \
#                 --gpu_id '1' \
#                 --batch_size 2 

# python train.py --model AttUResNeXt_class_v2_1 \
#                 --exp_id 9 \
#                 --lr 1e-4 \
#                 --lr_scheduler ReduceLROnPlateau \
#                 --dataset "Potsdam" \
#                 --num_classes 6 \
#                 --img_dir "/home/zj/data/potsdam/data_split/" \
#                 --label_dir "/home/zj/data/potsdam/label_split/" \
#                 --model_dir 'checkpoints/postdam' \
#                 --log_dir 'logs/potsdam' \
#                 --load_model_path '' \
#                 --gpu_id '1' \
#                 --batch_size 2 