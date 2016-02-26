#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import numpy
import argparse
import sklearn
import sklearn.neighbors

def cosdis(x, y):
    cosim = numpy.sum(x * y)
    cosim /= numpy.linalg.norm(x);
    cosim /= numpy.linalg.norm(y);
    
    return 1 - cosim 

def main(data_directory):
    train_set_x = numpy.load(os.path.join(data_directory, "train.feature.npy"))
    train_set_y = numpy.load(os.path.join(data_directory, "train.label.npy"))
    
    test_set_x = numpy.load(os.path.join(data_directory, "test.feature.npy"))
    test_set_y = numpy.load(os.path.join(data_directory, "test.label.npy"))
    
    for n_neighbors in [1, 3, 5, 10]:
        clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm='ball_tree', metric='pyfunc', func=cosdis)
        clf.fit(train_set_x, train_set_y)
        print "k-NN (k=%d, cos_dis)\t%f" % (n_neighbors, clf.score(test_set_x, test_set_y))
    
    '''
    for n_neighbors in [1, 3, 5, 10]:
        clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
        clf.fit(train_set_x, train_set_y)
        print "k-NN (k=%d, euclidean)\t%f" % (n_neighbors, clf.score(test_set_x, test_set_y))

    clf = sklearn.neighbors.NearestCentroid()
    clf.fit(train_set_x, train_set_y)
    print "nearest centroid\t%g" % clf.score(test_set_x, test_set_y)
    '''
    
if __name__ == "__main__":
    data_directory = sys.argv[1];
    main(data_directory)
