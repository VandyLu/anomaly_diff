
# export PYTHONUSERBASE=/newdata/fanbinlu/lib/anomaly
# export OPAL_PREFIX=/newdata/fanbinlu/openmpi-4.1.4/openmpi_dir/
# export PATH=$PATH:$OPAL_PREFIX/bin
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$OPAL_PREFIX/lib

MODEL_FLAGS="--image_size 128 --num_channels 128 --num_res_blocks 3 --learn_sigma True --resblock_updown True --use_scale_shift_norm True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"
TRAIN_FLAGS="--lr 1e-4 --batch_size 4 --save_interval 500 --lr_anneal_steps 2500 --ema_rate 0.995"


# MODEL_FLAGS="--image_size 128 --num_channels 128 --num_res_blocks 3 --learn_sigma True --resblock_updown True --use_scale_shift_norm True"
# DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --class_cond True"
# TRAIN_FLAGS="--lr 1e-4 --batch_size 4 --save_interval 5000 --lr_anneal_steps 20000 --ema_rate 0.997"


MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"
TRAIN_FLAGS="--lr 1e-4 --batch_size 2 --save_interval 10000 --lr_anneal_steps 100000 --ema_rate 0.997  --resume_checkpoint ./256x256_diffusion.pt"

name=$1
feat_shape=$2

export NCCL_P2P_DISABLE=1
# export OPENAI_LOGDIR=./work_dirs/uni256_pretrained_effnet_16x16_10k_1gpu
# mpiexec --oversubscribe -n 2 python ./scripts/image_train.py --data_dir ./data/MVTecAD/$name/train/ \
		# $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
# python ./scripts/image_train.py --data_dir ./data/UniMVTecAD/ \
# 		$MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
export OPENAI_LOGDIR=./work_dirs/Uni256_pretrained_effnet_feat64_10k_1gpu
python ./scripts/image_train.py --data_dir ./data/UniMVTecAD/ \
 	$MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS --feat_shape 64
# export OPENAI_LOGDIR=./work_dirs/${name}_256_pretrained_effnet_feat${feat_shape}_10k_1gpu
# python ./scripts/image_train.py --data_dir ./data/MVTecAD/$name/train/ \
 	# $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS --feat_shape $2

exit
# for name in  "bottle" "cable" "capsule" "carpet" "grid" "hazelnut" "leather" "metal_nut" "pill"  "screw" "tile" "toothbrush" "transistor" "wood" "zipper" 
for name in  "cable" "capsule" "carpet" "grid" 
do 
	echo $name
	# export OPENAI_LOGDIR=./work_dirs/${name}_128_2gpu
	# export OPENAI_LOGDIR=./work_dirs/${name}_256_10k_2gpu
	export OPENAI_LOGDIR=./work_dirs/${name}_256_pretrained_effnet_32x32_10k_1gpu
	python ./scripts/image_train.py --data_dir ./data/MVTecAD/$name/train/ \
 		$MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS --feat_shape 32
	# mpiexec --oversubscribe -n 2 python ./scripts/image_train.py --data_dir ./data/MVTecAD/$name/train/ \
		# $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS

	# mkdir visual
	
	# model=work_dirs/${name}_128_rot5_2gpu/ema_0.995_002500.pt

	# python ./scripts/image_anomaly.py --model_path $model \
	# $MODEL_FLAGS $DIFFUSION_FLAGS --timestep_respacing 100 --num_samples 100 \
	# --data_dir ./data/MVTecAD/${name}/test/ --train_data_dir ./data/MVTecAD/${name}/train/ \
	# --alpha_factor 1.0 --smooth True > result_${name}_rot5_smooth.txt
	
	# mv visual visual_rot5_${name}
	# mv *.pdf visual_rot5_${name}
	# model=work_dirs/${name}_5k_2gpu/ema_0.995_005000.pt
	# python ./scripts/image_anomaly.py --model_path $model \
	# 	$MODEL_FLAGS $DIFFUSION_FLAGS --timestep_respacing 100 --num_samples 100 \
	# 	--data_dir ./data/MVTecAD/$name/test/
	
	# mkdir $OPENAI_LOGDIR/anomaly_sample
	# mv *.jpg *.pdf $OPENAI_LOGDIR/anomaly_sample/

done

# export OPENAI_LOGDIR=./work_dirs/pill_5k_2gpu
# mpiexec -n 2 python ./scripts/image_train.py --data_dir './data/MVTecAD/pill/train/' \
# 	$MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
       
# export OPENAI_LOGDIR=./work_dirs/hazelnut_5k_2gpu
# mpiexec -n 2 python ./scripts/image_train.py --data_dir './data/MVTecAD/hazelnut/train/' \
# 	$MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
