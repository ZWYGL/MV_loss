#!/bin/bash
id=1
l2=0.0
l3=0.0  
mkdir -p ./experiment/metr/$id 
touch ./experiment/metr/$id/train-$id.log
CUDA_VISIBLE_DEVICES=0 python3 train.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj  --epoch 100 --expid $id  --save ./experiment/metr/$id --lambda_2 $l2 --lambda_3 $l3 > ./experiment/metr/$id/train-$id.log
