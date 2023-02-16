
#MODEL_FLAGS="--image_size 128 --num_channels 128 --num_res_blocks 3"
MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 3 --learn_sigma True --resblock_updown True --use_scale_shift_norm True"
MODEL_FLAGS="--image_size 128 --num_channels 128 --num_res_blocks 3 --learn_sigma True --resblock_updown True --use_scale_shift_norm True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"
TRAIN_FLAGS="--lr 1e-4 --batch_size 128"

name=wood
name=grid
name=toothbrush
name=zipper
name=metal_nut
name=screw
name=tile
name=carpet
name=leather
name=pill
name=hazelnut
name=transistor
name=bottle
name=cable
name=wood
name=capsule

dirname=${name}_128_samevar_2gpu

model=work_dirs_128_5k_2gpu/model005000.pt
model=work_dirs_128_5k_2gpu/ema_0.995_005000.pt
model=work_dirs/${name}_5k_2gpu/ema_0.995_005000.pt
model=work_dirs/${name}_64_2gpu/ema_0.995_002500.pt
model=work_dirs/${name}_128_20k_2gpu/ema_0.995_002500.pt
model=work_dirs/${dirname}/ema_0.995_002500.pt
# model=work_dirs/ema_0.995_005000.pt

# export OPENAI_LOGDIR=./work_dirs/${name}_128_20k_2gpu
export OPENAI_LOGDIR=./work_dirs/${dirname}

# python ./scripts/visual_loop.py --model_path $model \
# python ./scripts/image_sample.py --model_path $model \
# 	$MODEL_FLAGS $DIFFUSION_FLAGS --timestep_respacing 100 --num_samples 100 
	# --data_dir './data/MVTecAD/hazelnut/test/'

# python ./scripts/image_sample.py --model_path $model \
python ./scripts/image_anomaly.py --model_path $model \
	$MODEL_FLAGS $DIFFUSION_FLAGS --timestep_respacing 100 --num_samples 100 \
	--data_dir ./data/MVTecAD/${name}/test/ --train_data_dir ./data/MVTecAD/${name}/train/ \
	--alpha_factor 1.0
exit
	
# for name in "cable" "capsule" "hazelnut" "pill" "bottle" "wood" "toothbrush" "carpet" "grid" "leather" "metal_nut" "screw" "tile" "transistor" "zipper"
for name in "capsule" "hazelnut" "pill" 
do 
	echo $name
	# export OPENAI_LOGDIR=./work_dirs/${name}_128_2gpu
	export OPENAI_LOGDIR=./work_dirs/${name}_128_2gpu
	mkdir visual
	
	model=work_dirs/${name}_128_2gpu/ema_0.995_002500.pt

	python ./scripts/image_anomaly.py --model_path $model \
	$MODEL_FLAGS $DIFFUSION_FLAGS --timestep_respacing 100 --num_samples 100 \
	--data_dir ./data/MVTecAD/${name}/test/ --train_data_dir ./data/MVTecAD/${name}/train/ \
	--alpha_factor 1.0 --smooth True > result_${name}_smooth.txt
	
	mv visual visual_${name}
	mv *.pdf visual_${name}
done

# for name in "cable" "capsule" "hazelnut" "pill" "bottle" "wood" "toothbrush" "carpet" "grid" "leather" "metal_nut" "screw" "tile" "transistor" "zipper"
for name in "capsule" "hazelnut" "pill" 
do 
	echo $name
	# export OPENAI_LOGDIR=./work_dirs/${name}_128_2gpu
	export OPENAI_LOGDIR=./work_dirs/${name}_128_2gpu
	mkdir visual
	
	model=work_dirs/${name}_128_2gpu/ema_0.995_002500.pt

	python ./scripts/image_anomaly.py --model_path $model \
	$MODEL_FLAGS $DIFFUSION_FLAGS --timestep_respacing 100 --num_samples 100 \
	--data_dir ./data/MVTecAD/${name}/test/ --train_data_dir ./data/MVTecAD/${name}/train/ \
	--alpha_factor 1.0 --smooth False > result_${name}.txt
	
	mv visual visual_${name}
	mv *.pdf visual_${name}
done

