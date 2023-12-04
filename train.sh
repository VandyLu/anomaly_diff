
# 256 resolution, pretrained guided_diffusion UNet
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"
TRAIN_FLAGS="--lr 1e-4 --batch_size 2 --save_interval 5000 --lr_anneal_steps 10000 --ema_rate 0.997  --resume_checkpoint ./256x256_diffusion.pt"

dataset=$1
name=$2
feat_shape=$3

export NCCL_P2P_DISABLE=1
export OPENAI_LOGDIR=./work_dirs/${dataset}_${name}_pretrained_effnet_feat64_10k_1gpu

python ./scripts/image_train.py --data_dir ./data/$dataset/$name/train \
 	$MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS --feat_shape 64 --dataset_name $dataset
