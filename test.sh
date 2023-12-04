
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_scale_shift_norm True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"

name=$1
dirname=MVTecAD_${name}_pretrained_effnet_feat64_10k_1gpu
model=work_dirs/${dirname}/ema_0.997_010000.pt

export OPENAI_LOGDIR=./work_dirs/MVTecAD_${name}_pretrained_effnet_feat64_10k_1gpu_test

python ./scripts/image_anomaly.py --model_path $model \
	$MODEL_FLAGS $DIFFUSION_FLAGS --timestep_respacing 250 --num_samples 100 \
	--data_dir ./data/MVTecAD/${name}/test/ --train_data_dir ./data/MVTecAD/${name}/train/ \
	--alpha_factor 1.0 --beta_factor 1.0 --visual_dir visual/ --use_padim False --dataset_name MVTecAD --category $name --class_cond True
