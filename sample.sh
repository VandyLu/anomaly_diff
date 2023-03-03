
#MODEL_FLAGS="--image_size 128 --num_channels 128 --num_res_blocks 3"
MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 3 --learn_sigma True --resblock_updown True --use_scale_shift_norm True"
MODEL_FLAGS="--image_size 128 --num_channels 128 --num_res_blocks 3 --learn_sigma True --resblock_updown True --use_scale_shift_norm True"

# MODEL_FLAGS="--attention_resolutions 32,16,8 --image_size 128 --learn_sigma True --num_channels 256 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_scale_shift_norm True"
# MODEL_FLAGS="--image_size 256 --num_channels 128 --num_res_blocks 3 --learn_sigma True --resblock_updown True --use_scale_shift_norm True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"
TRAIN_FLAGS="--lr 1e-4 --batch_size 128"

name=toothbrush
name=tile
name=carpet
name=leather
name=pill
name=cable
name=metal_nut
name=screw
name=zipper
name=wood
name=bottle
name=transistor
name=grid
name=capsule
name=cable
name=hazelnut
name=wood

dirname=${name}_128_samevar_2gpu
dirname=${name}_256_10k_2gpu
dirname=${name}_128_rot5_2gpu
dirname=uni128_20k_1gpu
dirname=uni128_pretrain_20k_1gpu

model=work_dirs_128_5k_2gpu/model005000.pt
model=work_dirs_128_5k_2gpu/ema_0.995_005000.pt
model=work_dirs/${name}_5k_2gpu/ema_0.995_005000.pt
model=work_dirs/${name}_64_2gpu/ema_0.995_002500.pt
model=work_dirs/${name}_256_10k_2gpu/ema_0.995_010000.pt
model=work_dirs/${dirname}/ema_0.995_002500.pt
model=work_dirs/${dirname}/ema_0.997_020000.pt

class_cond=False
# model=work_dirs/${dirname}/ema_0.995_010000.pt
# model=work_dirs/ema_0.995_005000.pt

# export OPENAI_LOGDIR=./work_dirs/${name}_128_20k_2gpu
export OPENAI_LOGDIR=./work_dirs/${dirname}

# python ./scripts/image_sample.py --model_path $model \
# 	$MODEL_FLAGS $DIFFUSION_FLAGS --timestep_respacing 250 --num_samples 50 
# exit

# python ./scripts/image_anomaly.py --model_path $model \
# 	$MODEL_FLAGS $DIFFUSION_FLAGS --timestep_respacing 250 --num_samples 100 \
# 	--data_dir ./data/MVTecAD/${name}/test/ --train_data_dir ./data/MVTecAD/${name}/train/ \
# 	--alpha_factor 1.0 --visual_dir visual/ --use_padim False --category $name --class_cond $class_cond
# 	# --data_dir ./data/MVTecAD/simple --train_data_dir ./data/MVTecAD/${name}/train/ \
# 	# --alpha_factor 1.0 --visual_dir visual/ --use_padim False --category $name
# exit
# # cut_inner_insulation
# python ./scripts/guided_sample.py --model_path $model \
# 	$MODEL_FLAGS $DIFFUSION_FLAGS --timestep_respacing 250 --num_samples 100 \
# 	--data_dir ./data/MVTecAD/${name}/test/thread --train_data_dir ./data/MVTecAD/${name}/train/ \
# 	--alpha_factor 1.0 --category $name
# 	# --data_dir ./data/MVTecAD/simple/test --train_data_dir ./data/MVTecAD/${name}/train/ \
# 	# --alpha_factor 1.0 --visual_dir visual/ --use_padim False
# exit
	
# for name in "hazelnut" 
for name in  "bottle" "cable" "capsule" "carpet" "grid" "hazelnut" "leather" "metal_nut" "pill"  "screw" "tile" "toothbrush" "transistor" "wood" "zipper" 
do 
	echo $name
	# dirname=${name}_128_rot5_2gpu
	# export OPENAI_LOGDIR=./work_dirs/$dirname

	# visual_dir=${name}_final_visual
	# mkdir $visual_dir
	
	# model=work_dirs/${dirname}/ema_0.995_002500.pt
	# python ./scripts/image_anomaly.py --model_path $model \
	# $MODEL_FLAGS $DIFFUSION_FLAGS --timestep_respacing 250 --num_samples 100 \
	# --data_dir ./data/MVTecAD/${name}/test/ --train_data_dir ./data/MVTecAD/${name}/train/ \
	# --alpha_factor 1.0 --smooth False --use_padim False --category $name

	dirname=${name}_128_rot5_2gpu
	model=work_dirs/${dirname}/ema_0.995_002500.pt
	python ./scripts/image_anomaly.py --model_path $model \
	$MODEL_FLAGS $DIFFUSION_FLAGS --timestep_respacing 250 --num_samples 100 \
	--data_dir ./data/MVTecAD/${name}/test/ --train_data_dir ./data/MVTecAD/${name}/train/ \
	--alpha_factor 1.0 --visual_dir visual/ --use_padim False --category $name --class_cond $class_cond
	
	# mv *.pdf $visual_dir
done
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
	--alpha_factor 1.0 --smooth False > result_${name}.txt
	
	mv visual visual_${name}
	mv *.pdf visual_${name}
done

