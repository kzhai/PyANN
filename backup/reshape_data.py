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

def test(input_directory, output_directory):
    # flat_images = scipy.load("/Users/kzhai/Workspace/PyANN/input/mnist_784/test.label.npy")
    # print flat_images.shape
    input_file = os.path.join(input_directory, "test.feature.npy")
    flat_images = scipy.load(input_file);
    print flat_images.shape;
    number_of_data = flat_images.shape[0];
    reshaped_images = numpy.reshape(flat_images, (number_of_data, 1, 28, 28))
    print reshaped_images.shape;
    output_file = os.path.join(output_directory, "test.feature.npy")    
    numpy.save(output_file, reshaped_images)
    
    input_file = os.path.join(input_directory, "train.feature.npy")
    flat_images = scipy.load(input_file);
    print flat_images.shape;
    number_of_data = flat_images.shape[0];
    reshaped_images = numpy.reshape(flat_images, (number_of_data, 1, 28, 28))
    print reshaped_images.shape;
    output_file = os.path.join(output_directory, "train.feature.npy")    
    numpy.save(output_file, reshaped_images)
    
    '''
    images_for_display = reshaped_images
    matplotlib.pyplot.imshow(images_for_display[0, 0, :, :])
    matplotlib.pyplot.show()
    matplotlib.pyplot.imshow(images_for_display[-1, 0, :, :])
    matplotlib.pyplot.show()
    '''
    
if __name__ == '__main__':
    input_directory = "/Users/kzhai/Workspace/PyANN/input/mnist_784";
    output_directory = "/Users/kzhai/Workspace/PyANN/input/mnist_1x28x28";
    
    test(input_directory, output_directory)
