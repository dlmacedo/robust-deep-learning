#!/usr/bin/env bash

python train.py --dir all --dataset cifar10 --model densenetbc100 --loss softmax_none -x 10 --gpu 0
python calibrate.py --dir all --dataset cifar10 --model densenetbc100 --loss softmax_none -x 10 --gpu 0
python detect.py --dir all --dataset cifar10 --model densenetbc100 --loss softmax_none -x 10 --gpu 0

python train.py --dir all --dataset cifar10 --model densenetbc100 --loss isomax_none -x 10 --gpu 0
python calibrate.py --dir all --dataset cifar10 --model densenetbc100 --loss isomax_none -x 10 --gpu 0
python detect.py --dir all --dataset cifar10 --model densenetbc100 --loss isomax_none -x 10 --gpu 0

python train.py --dir all --dataset cifar10 --model densenetbc100 --loss isomaxplus_none -x 10 --gpu 0
python calibrate.py --dir all --dataset cifar10 --model densenetbc100 --loss isomaxplus_none -x 10 --gpu 0
python detect.py --dir all --dataset cifar10 --model densenetbc100 --loss isomaxplus_none -x 10 --gpu 0

python train.py --dir all --dataset cifar10 --model densenetbc100 --loss dismax_none -x 10 --gpu 0
python calibrate.py --dir all --dataset cifar10 --model densenetbc100 --loss dismax_none -x 10 --gpu 0
python detect.py --dir all --dataset cifar10 --model densenetbc100 --loss dismax_none -x 10 --gpu 0

python train.py --dir all --dataset cifar10 --model densenetbc100 --loss dismax_fpr -x 10 --gpu 0
python calibrate.py --dir all --dataset cifar10 --model densenetbc100 --loss dismax_fpr -x 10 --gpu 0
python detect.py --dir all --dataset cifar10 --model densenetbc100 --loss dismax_fpr -x 10 --gpu 0
