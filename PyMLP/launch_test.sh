#!/bin/bash

PYANN_HOME=$HOME/Workspace/PyANN

if [ $# != 2 ]; then
    echo "usage: launch_test.sh INPUT_DIRECTORY OUTPUT_DIRECTORY"
    exit
fi

INPUT_DIRECTORY=$(grealpath $1)
OUTPUT_DIRECTORY=$(grealpath $2)

for MODEL_DIRECTORY in $OUTPUT_DIRECTORY/*
do
	if [ -f "$MODEL_DIRECTORY" ]; then
		continue
    fi
	
	if [ -f "$MODEL_DIRECTORY/model.pkl" ]; then
		echo "Evaluating $MODEL_DIRECTORY"
    	python -um PyMLP.launch_test --input_directory=$INPUT_DIRECTORY --model_directory=$MODEL_DIRECTORY --best_model_only
	fi
done