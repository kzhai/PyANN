#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import numpy
import argparse

import sklearn
import sklearn.svm

def main(data_directory):
    train_set_x = numpy.load(os.path.join(data_directory, "train.feature.npy"))
    train_set_y = numpy.load(os.path.join(data_directory, "train.label.npy"))
    
    test_set_x = numpy.load(os.path.join(data_directory, "test.feature.npy"))
    test_set_y = numpy.load(os.path.join(data_directory, "test.label.npy"))
    
    #clf = sklearn.svm.SVC(kernel="linear")
    #clf.fit(train_set_x, train_set_y)
    #print "SVC+linear", clf.score(test_set_x, test_set_y)
    
    clf = sklearn.svm.SVC(kernel="rbf")
    clf.fit(train_set_x, train_set_y)
    print "SVM (RBF kernel)\t%f" % clf.score(test_set_x, test_set_y)
    
    clf = sklearn.svm.LinearSVC()
    clf.fit(train_set_x, train_set_y)
    print "SVM (linear kernel)\t%f" % clf.score(test_set_x, test_set_y)
    
if __name__ == "__main__":
    data_directory = sys.argv[1];
    main(data_directory)
