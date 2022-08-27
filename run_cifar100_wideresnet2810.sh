#!/usr/bin/env bash

python train.py --dir all --dataset cifar100 --model wideresnet2810 --loss softmax_none -x 10 --gpu 0
python calibrate.py --dir all --dataset cifar100 --model wideresnet2810 --loss softmax_none -x 10 --gpu 0
python detect.py --dir all --dataset cifar100 --model wideresnet2810 --loss softmax_none -x 10 --gpu 0

python train.py --dir all --dataset cifar100 --model wideresnet2810 --loss isomax_none -x 10 --gpu 0
python calibrate.py --dir all --dataset cifar100 --model wideresnet2810 --loss isomax_none -x 10 --gpu 0
python detect.py --dir all --dataset cifar100 --model wideresnet2810 --loss isomax_none -x 10 --gpu 0

python train.py --dir all --dataset cifar100 --model wideresnet2810 --loss isomaxplus_none -x 10 --gpu 0
python calibrate.py --dir all --dataset cifar100 --model wideresnet2810 --loss isomaxplus_none -x 10 --gpu 0
python detect.py --dir all --dataset cifar100 --model wideresnet2810 --loss isomaxplus_none -x 10 --gpu 0

python train.py --dir all --dataset cifar100 --model wideresnet2810 --loss dismax_none -x 10 --gpu 0
python calibrate.py --dir all --dataset cifar100 --model wideresnet2810 --loss dismax_none -x 10 --gpu 0
python detect.py --dir all --dataset cifar100 --model wideresnet2810 --loss dismax_none -x 10 --gpu 0

python train.py --dir all --dataset cifar100 --model wideresnet2810 --loss dismax_fpr -x 10 --gpu 0
python calibrate.py --dir all --dataset cifar100 --model wideresnet2810 --loss dismax_fpr -x 10 --gpu 0
python detect.py --dir all --dataset cifar100 --model wideresnet2810 --loss dismax_fpr -x 10 --gpu 0
