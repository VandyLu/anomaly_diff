MODEL_FLAGS="--image_size 128 --num_channels 128 --num_res_blocks 3 --learn_sigma True --resblock_updown True --use_scale_shift_norm True"

# python ./scripts/image_train.py --data_dir './data/MVTecAD/cable/train/' \

name=capsule
name=bottle
name=capsule
name=grid
name=toothbrush
name=zipper
name=metal_nut
name=screw
name=tile
name=transistor
name=carpet
name=leather
name=pill
name=hazelnut
name=cable


# for name in  "bottle" "cable" "capsule" "carpet" "grid" "hazelnut" "leather" "metal_nut" "pill"  "screw" "tile" "toothbrush" "transistor" "wood" "zipper" 
for name in "cable"
do
	# cat result_${name}_rot5_smooth.txt |grep pro |head -n 1 
	# echo $name
	python ./scripts/padim.py --data_dir ./data/MVTecAD/$name/test/ \
		--train_data_dir ./data/MVTecAD/$name/train --category $name --image_size 256
done
exit

export NCCL_P2P_DISABLE=1
export OPENAI_LOGDIR=./work_dirs/${name}_128_2gpu
# for name in "capsule"
for name in "cable" "capsule" "pill" "bottle" "wood" "toothbrush" "carpet" "grid" "leather" "metal_nut" "screw" "tile" "transistor" "zipper" "hazelnut"
do
	echo $name
	python ./scripts/padim.py --data_dir ./data/MVTecAD/$name/test/ --train_data_dir ./data/MVTecAD/$name/train --category $name
done

exit
for name in "bottle" "wood" "toothbrush" "carpet" "grid" "leather" "metal_nut" "screw" "tile" "transistor" "zipper"
# for name in "capsule"
do 
	echo $name
	# export OPENAI_LOGDIR=./work_dirs/${name}_5k_2gpu
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
