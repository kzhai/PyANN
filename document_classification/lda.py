#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import numpy
import argparse

import sklearn
import sklearn.svm
import sklearn.decomposition

def main(data_directory):
    train_set_x = numpy.load(os.path.join(data_directory, "train.feature.npy"))
    train_set_y = numpy.load(os.path.join(data_directory, "train.label.npy"))
    
    test_set_x = numpy.load(os.path.join(data_directory, "test.feature.npy"))
    test_set_y = numpy.load(os.path.join(data_directory, "test.label.npy"))
    
    joint_set = numpy.vstack((train_set_x, test_set_x));
    
    for n_topics in [10, 20, 50, 100]:
        lda = sklearn.decomposition.LatentDirichletAllocation(n_topics)
        lda.fit(joint_set)
        train_doc_topic_dist = lda.transform(train_set_x)
        test_doc_topic_dist = lda.transform(test_set_x)
        
        clf = sklearn.svm.LinearSVC()
        clf.fit(train_doc_topic_dist, train_set_y)
        print "LDA (k=%d) + SVM\t%f" % (n_topics, clf.score(test_doc_topic_dist, test_set_y))
    
if __name__ == "__main__":
    data_directory = sys.argv[1];
    main(data_directory)
