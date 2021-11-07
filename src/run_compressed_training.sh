#!/bin/bash

# Use this as a template to perform quantisation and pruning at different bit widths, the commands here are specific to CERN's servers. 
# This cript spawns 4 shells on 2 different machines to run each segment on a different GPU.

# Arguements for this script
# Argument 1: first machine
# Argument 2: second machine

run() {
    for i in $1 $2; do python train.py --model cnn --signals "./signals_old/*" --dataset dataset_old/dataset.h5 --outdir ./output/cnn_qp$i --quant_size $i --pruning True --device $3; done
}

ssh "$1" "`run 2 4 0`" &
ssh "$1" "`run 6 8 1`" &
ssh "$2" "`run 10 12 0`" &
ssh "$2" "`run 14 16 1`" &