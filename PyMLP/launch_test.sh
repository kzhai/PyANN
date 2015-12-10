#!/bin/bash

PYANN_HOME=$HOME/Workspace/PyANN

if [ $# == 2 ]; then
    INPUT_DIRECTORY=$1
    OUTPUT_DIRECTORY=$2
    BATCH_SIZE=100
    GPU_DEVICE=gpu
elif [ $# == 3 ]; then
    INPUT_DIRECTORY=$1
    OUTPUT_DIRECTORY=$2
    BATCH_SIZE=$3
    GPU_DEVICE=gpu
elif [ $# == 4 ]; then
    INPUT_DIRECTORY=$1
    OUTPUT_DIRECTORY=$2
    BATCH_SIZE=$3
    GPU_DEVICE=$4
else
    echo "usage: launch_test.sh INPUT_DIRECTORY OUTPUT_DIRECTORY [BATCH_SIZE] [GPU_DEVICE]"
    exit
fi

for MODEL_DIRECTORY in $OUTPUT_DIRECTORY/*
do
    if [ -f "$MODEL_DIRECTORY" ]; then
	continue
    fi
	
    if [ -f "$MODEL_DIRECTORY/model.pkl" ]; then
	echo "Evaluating $MODEL_DIRECTORY"
    	THEANO_FLAGS=mode=FAST_RUN,device=$GPU_DEVICE,floatX=float32 python -um PyMLP.launch_test --input_directory=$INPUT_DIRECTORY --model_directory=$MODEL_DIRECTORY --batch_size=$BATCH_SIZE --best_model_only
    fi
done