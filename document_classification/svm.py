#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import numpy
import argparse
from collections import Counter
from scipy.sparse import csr_matrix
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

import sklearn
import sklearn.svm

def main(data_directory, kernel="linear"):
    train_set_x = numpy.load(os.path.join(data_directory, "train.feature.npy"))
    train_set_y = numpy.load(os.path.join(data_directory, "train.label.npy"))
    
    test_set_x = numpy.load(os.path.join(data_directory, "test.feature.npy"))
    test_set_y = numpy.load(os.path.join(data_directory, "test.label.npy"))
    
    clf = sklearn.svm.SVC(kernel=kernel)
    clf.fit(train_set_x, train_set_y)
    print clf.score(test_set_x, test_set_y)
    
if __name__ == "__main__":
    data_directory = sys.argv[1];
    main(data_directory)
