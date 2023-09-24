#!/bin/sh



(
cd ../


CUDA_VISIBLE_DEVICES=0 python train_script.py \
	--img-dir=$IMGDIR \
	--restore-from=$RESTORE \
	--train-list=${TRAIN:=Fungal_train.txt} \
  --num-workers=4 \
	--model=${MODEL:=resnet50} \
  --version="${VERSION:="TAME"}" \
	--layers="${LAYERS:="layer2 layer3 layer4"}" \
  --wd=${WD:=5e-4} \
	--max-lr=${MLR:=1e-2} \
	--epoch=${EPOCHS:=8} \
	--batch-size=${BSIZE:=32}


CUDA_VISIBLE_DEVICES=0 python eval_script.py \
	--val-dir=$VALDIR \
  --restore-from=$RESTORE \
	--test-list=${TEST:="Fungal_eval.txt"} \
	--num-workers=4 \
  --model=${MODEL:="resnet50"} \
  --version=${VERSION:="TAME"} \
	--layers="${LAYERS:="layer2 layer3 layer4"}" \
	--start-epoch=1 \
	--end-epoch=${EPOCHS:=32}


exit 0
)
