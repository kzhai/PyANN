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

def preprocess_directory(input_directory):
    input_train_file = os.path.join(input_directory, "train.feature.npy")
    
    input_train_data = scipy.load(input_train_file);
    # input_train_data = input_train_data / numpy.float32(256);
    
    print input_train_data.dtype, input_train_data.shape, numpy.max(input_train_data), numpy.min(input_train_data);
    
    input_test_file = os.path.join(input_directory, "test.feature.npy")

    input_test_data = scipy.load(input_test_file);
    
    print input_test_data.dtype, input_test_data.shape, numpy.max(input_test_data), numpy.min(input_test_data);
    
if __name__ == '__main__':
    input_directory = sys.argv[1]
    
    preprocess_directory(input_directory)
