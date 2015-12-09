import os
import sys
import timeit

import cPickle
import numpy
import scipy
import scipy.io

import theano
import theano.tensor

import time
import datetime
import optparse

import lasagne

import matplotlib
import matplotlib.pyplot

def parse_svhn(input_directory, output_directory):
    labels_for_train = numpy.zeros(0, dtype=numpy.uint8);
    images_for_train = numpy.zeros((0, 3, 32, 32), dtype=numpy.uint8);
    
    for file_name in ["train_32x32.mat", "extra_32x32.mat"]:
        input_file = os.path.join(input_directory, file_name);
        a = scipy.io.loadmat(input_file)
        train_labels = a["y"]
        train_images = a["X"]
        
        train_labels = train_labels[:, 0]
        train_labels[train_labels == 10] = 0;

        train_images = numpy.swapaxes(train_images, 2, 3)
        train_images = numpy.swapaxes(train_images, 0, 2)
        train_images = numpy.swapaxes(train_images, 1, 3)
        
        labels_for_train = numpy.concatenate((labels_for_train, train_labels), axis=0);
        images_for_train = numpy.concatenate((images_for_train, train_images), axis=0);
        
    print numpy.max(labels_for_train), numpy.min(labels_for_train)
    print numpy.max(images_for_train), numpy.min(images_for_train)    
    print labels_for_train.shape
    print images_for_train.shape
    
    output_file = os.path.join(output_directory, "train.feature.npy")    
    numpy.save(output_file, images_for_train)
    output_file = os.path.join(output_directory, "train.label.npy")    
    numpy.save(output_file, labels_for_train)
    
    images_for_display = numpy.swapaxes(images_for_train, 1, 2)
    images_for_display = numpy.swapaxes(images_for_display, 2, 3)
    matplotlib.pyplot.imshow(images_for_display[0, :, :, :])
    matplotlib.pyplot.show()
    matplotlib.pyplot.imshow(images_for_display[-1, :, :, :])
    matplotlib.pyplot.show()
    
    labels_for_test = numpy.zeros(0, dtype=numpy.uint8);
    images_for_test = numpy.zeros((0, 3, 32, 32), dtype=numpy.uint8);
    for file_name in ["test_32x32.mat"]:
        input_file = os.path.join(input_directory, file_name);
        a = scipy.io.loadmat(input_file)
        test_labels = a["y"]
        test_images = a["X"]
        
        test_labels = test_labels[:, 0]
        test_labels[test_labels == 10] = 0;
        
        test_images = numpy.swapaxes(test_images, 2, 3)
        test_images = numpy.swapaxes(test_images, 0, 2)
        test_images = numpy.swapaxes(test_images, 1, 3)
        
        labels_for_test = numpy.concatenate((labels_for_test, test_labels), axis=0);
        images_for_test = numpy.concatenate((images_for_test, test_images), axis=0);
    
    output_file = os.path.join(output_directory, "test.feature.npy")    
    numpy.save(output_file, images_for_test)
    output_file = os.path.join(output_directory, "test.label.npy")    
    numpy.save(output_file, labels_for_test)

    print numpy.bincount(labels_for_test)
    print labels_for_test.shape
    print images_for_test.shape
    
    images_for_display = numpy.swapaxes(images_for_test, 1, 2)
    images_for_display = numpy.swapaxes(images_for_display, 2, 3)
    matplotlib.pyplot.imshow(images_for_display[0, :, :, :])
    matplotlib.pyplot.show()
    matplotlib.pyplot.imshow(images_for_display[-1, :, :, :])
    matplotlib.pyplot.show()
        
if __name__ == '__main__':
    input_directory = sys.argv[1];
    output_directory = sys.argv[2]
    parse_svhn(input_directory, output_directory)
