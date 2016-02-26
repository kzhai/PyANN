#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import numpy

def main(data_directory):
    train_set_x = numpy.load(os.path.join(data_directory, "train.feature.npy"))
    train_set_y = numpy.load(os.path.join(data_directory, "train.label.npy"))
    
    test_set_x = numpy.load(os.path.join(data_directory, "test.feature.npy"))
    test_set_y = numpy.load(os.path.join(data_directory, "test.label.npy"))
    
    train_set_x_over_norm = train_set_x / numpy.linalg.norm(train_set_x, axis=1)[:, numpy.newaxis];
    assert train_set_x_over_norm.shape == train_set_x.shape
    
    cosine_similarity = numpy.dot(test_set_x, train_set_x_over_norm.T);
    most_similar_index = numpy.argmax(cosine_similarity, axis=1);
    predicted_label = train_set_y[most_similar_index]
    print 1.0 * numpy.sum(predicted_label == test_set_y) / len(predicted_label)

if __name__ == "__main__":
    data_directory = sys.argv[1];
    main(data_directory)
