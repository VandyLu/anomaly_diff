MODEL_FLAGS="--image_size 256 --num_channels 128 --num_res_blocks 3 --learn_sigma True --resblock_updown True --use_scale_shift_norm True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"
TRAIN_FLAGS="--lr 1e-4 --batch_size 2 --save_interval 500 --lr_anneal_steps 2500 --ema_rate 0.995"
#TRAIN_FLAGS="--lr 1e-4 --batch_size 2 --save_interval 1000 --lr_anneal_steps 5000 --ema_rate 0.995 --resume_checkpoint ./128x128_diffusion.pt"

# python ./scripts/image_train.py --data_dir './data/MVTecAD/cable/train/' \

name=capsule
export NCCL_P2P_DISABLE=1
export OPENAI_LOGDIR=./work_dirs/${name}_256_4gpu
mpiexec --oversubscribe -n 4 python ./scripts/image_train.py --data_dir ./data/MVTecAD/$name/train/ \
		$MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS

exit
# for name in "bottle" "wood" "toothbrush" "carpet" "grid" "leather" "metal_nut" "screw" "tile" "transistor" "zipper"
for name in "capsule"
do 
	echo $name
	export OPENAI_LOGDIR=./work_dirs/${name}_5k_2gpu
	mpiexec -n 2 python ./scripts/image_train.py --data_dir ./data/MVTecAD/$name/train/ \
		$MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
	
	model=work_dirs/${name}_5k_2gpu/ema_0.995_005000.pt
	python ./scripts/image_anomaly.py --model_path $model \
		$MODEL_FLAGS $DIFFUSION_FLAGS --timestep_respacing 100 --num_samples 100 \
		--data_dir ./data/MVTecAD/$name/test/
	
	mkdir $OPENAI_LOGDIR/anomaly_sample
	mv *.jpg *.pdf $OPENAI_LOGDIR/anomaly_sample/

done

# export OPENAI_LOGDIR=./work_dirs/pill_5k_2gpu
# mpiexec -n 2 python ./scripts/image_train.py --data_dir './data/MVTecAD/pill/train/' \
# 	$MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
       
# export OPENAI_LOGDIR=./work_dirs/hazelnut_5k_2gpu
# mpiexec -n 2 python ./scripts/image_train.py --data_dir './data/MVTecAD/hazelnut/train/' \
# 	$MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
