#!/bin/bash
id=32
l2=1.0
l3=0.01
l4=1.0
mkdir -p ./experiment/metr/$id 
touch ./experiment/metr/$id/train-$id.log
CUDA_VISIBLE_DEVICES=3 python3 train.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj  --epoch 100 --expid $id  --save ./experiment/metr/$id --lambda_2 $l2 --lambda_3 $l3 --lambda_4 $l4 --freeze 1 --start 1 --end 5 > ./experiment/metr/$id/train-$id.log
