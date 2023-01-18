
#MODEL_FLAGS="--image_size 128 --num_channels 128 --num_res_blocks 3"
MODEL_FLAGS="--image_size 256 --num_channels 128 --num_res_blocks 3 --learn_sigma True --resblock_updown True --use_scale_shift_norm True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"
TRAIN_FLAGS="--lr 1e-4 --batch_size 128"

name=capsule
model=work_dirs_128_5k_2gpu/model005000.pt
model=work_dirs_128_5k_2gpu/ema_0.995_005000.pt
model=work_dirs/${name}_5k_2gpu/ema_0.995_005000.pt
model=work_dirs/capsule_256_4gpu/ema_0.995_002500.pt
# model=work_dirs/ema_0.995_005000.pt

export OPENAI_LOGDIR=./work_dirs/${name}_256_4gpu

# python ./scripts/visual_loop.py --model_path $model \
# python ./scripts/image_sample.py --model_path $model \
# 	$MODEL_FLAGS $DIFFUSION_FLAGS --timestep_respacing 100 --num_samples 100 
	# --data_dir './data/MVTecAD/hazelnut/test/'

# python ./scripts/image_sample.py --model_path $model \
python ./scripts/image_anomaly.py --model_path $model \
	$MODEL_FLAGS $DIFFUSION_FLAGS --timestep_respacing 100 --num_samples 100 \
	--data_dir ./data/MVTecAD/${name}/test/
	

