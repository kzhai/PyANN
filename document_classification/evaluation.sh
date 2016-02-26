#!/bin/bash

PYANN_HOME=$HOME/Workspace/PyANN

INPUT_DIRECTORY=$1
echo "python document_classification/cosim.py $INPUT_DIRECTORY\\_normalized"

python document_classification/cosim.py $INPUT_DIRECTORY\_normalized
python document_classification/knn.py $INPUT_DIRECTORY\_normalized
python document_classification/svm.py $INPUT_DIRECTORY\_normalized

python document_classification/nb.py $INPUT_DIRECTORY
python document_classification/lda.py $INPUT_DIRECTORY

python document_classification/cosim.py $INPUT_DIRECTORY
python document_classification/knn.py $INPUT_DIRECTORY
python document_classification/svm.py $INPUT_DIRECTORY
