import collections;
import cPickle
import os;
import operator;
import math;
import numpy;
import numpy.random;
import scipy;
import scipy.linalg;
import sklearn
import sklearn.covariance
import sys
import time;
import theano
import theano.tensor
import matplotlib
import matplotlib.pyplot

def standard_normalization(input_data, data_mean, data_std):
    input_data -= data_mean
    input_data /= data_std
    return input_data

def preprocess_directory(input_directory, output_directory):
    input_train_file = os.path.join(input_directory, "train.feature.npy")
    output_train_file = os.path.join(output_directory, "train.feature.npy")
    
    input_train_data = scipy.load(input_train_file);
    
    input_train_data = input_train_data.astype(numpy.float32)
    input_train_data /= numpy.sum(input_train_data, axis=1)[:, numpy.newaxis];
    
    print input_train_data.dtype, input_train_data.shape, numpy.max(input_train_data), numpy.min(input_train_data);
    
    numpy.save(output_train_file, input_train_data)
    
    input_test_file = os.path.join(input_directory, "test.feature.npy")
    output_test_file = os.path.join(output_directory, "test.feature.npy")
    
    input_test_data = scipy.load(input_test_file);
    
    input_test_data = input_test_data.astype(numpy.float32)
    input_test_data /= numpy.sum(input_test_data, axis=1)[:, numpy.newaxis];
    
    print input_test_data.dtype, input_test_data.shape, numpy.max(input_test_data), numpy.min(input_test_data);
    
    numpy.save(output_test_file, input_test_data)
    
if __name__ == '__main__':
    input_directory = sys.argv[1]
    output_directory = sys.argv[2]
    
    preprocess_directory(input_directory, output_directory)
