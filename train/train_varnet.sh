#!/bin/bash

datadir='/mnt/dense/kanghyun/Brain_T1Post'
acc=4
cuda=2
experimentname='Brain_T1Post_R4'

# Example running code:
# Train Brain_T1Post, R=4, experiment name: Brain_T1Post_R4, cuda:2
# bash train_varnet.sh -d /mnt/dense/kanghyun/Brain_T1Post -r 4 -e Brain_T1Post_R4 -c 2

# Train Knee_PD, R=4, experiment name: KneePD_R4, cuda:1
# bash train_varnet.sh -d /mnt/dense/kanghyun/fastMRI_mini_PD -r 4 -e Knee_PD_R4 -c 1

while getopts d:r:e:c:h flag; do
    case $flag in 
    d) 
        datadir=$OPTARG
        ;;
    r)
        acc=$OPTARG        
        ;;
    e)
        experimentname=$OPTARG
        ;;
    c)
        cuda=$OPTARG
        ;;
    h)
        echo "Bash script that launches Training Variational Network"
        echo ""
        echo "Usage: $0 -d <data-dir> -r <acceleration> -e <experiment name>"
        echo ""
        echo "2021 Kanghyun Ryu <kanghyun@stanford.edu>"
        exit
        ;;

    \?)
        exit 1
        ;;
    esac
done

echo "datadir: $datadir"
echo "acc: $acc"
echo "experimentname: $experimentname"
echo "cuda: $cuda"

CUDA_VISIBLE_DEVICES=$cuda python /home/kanghyun/projRSC/ESPIRIT_RSC/train/train_varnet.py --datadir $datadir --num-epochs 200 --R $acc --experimentname $experimentname



