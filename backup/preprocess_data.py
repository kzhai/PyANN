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

def preprocess_file(input_file, output_file):
    #input_file = os.path.join(input_directory, "test.feature.npy")
    #output_file = os.path.join(output_directory, "test.feature.npy")
    
    input_data = scipy.load(input_file);
    # input_data = input_data / numpy.float32(256);
    
    input_data = input_data / numpy.float32(1.0);
    input_data -= numpy.mean(input_data);
    input_data /= numpy.std(input_data);
    
    print input_file
    print output_file
    print input_data.dtype, numpy.max(input_data), numpy.min(input_data);
    
    numpy.save(output_file, input_data)

def preprocess_directory(input_directory, output_directory):
    input_file = os.path.join(input_directory, "train.feature.npy")
    output_file = os.path.join(output_directory, "train.feature.npy")
    preprocess_file(input_file, output_file)
    
    input_file = os.path.join(input_directory, "test.feature.npy")
    output_file = os.path.join(output_directory, "test.feature.npy")
    preprocess_file(input_file, output_file)
    
if __name__ == '__main__':
    input_directory = sys.argv[1]
    output_directory = sys.argv[2]
    
    preprocess_directory(input_directory, output_directory)
