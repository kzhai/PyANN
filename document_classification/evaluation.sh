#!/bin/bash

PYANN_HOME=$HOME/Workspace/PyANN

INPUT_DIRECTORY=$1

python document_classification/cosim.py $INPUT_DIRECTORY_normalized
wait
python document_classification/knn.py $INPUT_DIRECTORY_normalized
wait
python document_classification/svm.py $INPUT_DIRECTORY_normalized
wait

python document_classification/nb.py $INPUT_DIRECTORY
wait
python document_classification/lda $INPUT_DIRECTORY
wait

python document_classification/cosim.py $INPUT_DIRECTORY
wait
python document_classification/knn.py $INPUT_DIRECTORY
wait
python document_classification/svm.py $INPUT_DIRECTORY
wait