#!/usr/bin/env bash

#python train.py --dir all --dataset imagenet1k --model resnet18 --loss softmax_none --gpu 0 -bs 256 -e 90 -lrde "30 60"
#python calibrate.py --dir all --dataset imagenet1k --model resnet18 --loss softmax_none --gpu 0
#python detect.py --dir all --dataset imagenet1k --model resnet18 --loss softmax_none --gpu 0

#python train.py --dir all --dataset imagenet1k --model resnet18 --loss isomax_none --gpu 0 -bs 256 -e 90 -lrde "30 60"
#python calibrate.py --dir all --dataset imagenet1k --model resnet18 --loss isomax_none --gpu 0
#python detect.py --dir all --dataset imagenet1k --model resnet18 --loss isomax_none --gpu 0

#python train.py --dir all --dataset imagenet1k --model resnet18 --loss isomaxplus_none --gpu 0 -bs 256 -e 90 -lrde "30 60"
#python calibrate.py --dir all --dataset imagenet1k --model resnet18 --loss isomaxplus_none --gpu 0
#python detect.py --dir all --dataset imagenet1k --model resnet18 --loss isomaxplus_none --gpu 0

#python train.py --dir all --dataset imagenet1k --model resnet18 --loss dismax_none --gpu 0 -bs 256 -e 90 -lrde "30 60"
#python calibrate.py --dir all --dataset imagenet1k --model resnet18 --loss dismax_none --gpu 0
#python detect.py --dir all --dataset imagenet1k --model resnet18 --loss dismax_none --gpu 0

##python train.py --dir all --dataset imagenet1k --model resnet18 --loss dismax_fpr --gpu 0 -bs 256 -e 90 -lrde "30 60"
python calibrate.py --dir all --dataset imagenet1k --model resnet18 --loss dismax_fpr --gpu 0
##python detect.py --dir all --dataset imagenet1k --model resnet18 --loss dismax_fpr --gpu 0
