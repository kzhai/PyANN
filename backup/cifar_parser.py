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

def cifar_parser(input_directory, output_directory):
    input_train_feature_file = os.path.join(input_directory, "train.feature.npy")
    input_train_feature = scipy.load(input_train_feature_file);
    input_train_feature = numpy.reshape(input_train_feature, (input_train_feature.shape[0], 3, 32, 32))
    # print input_train_feature.dtype, input_train_feature.shape, numpy.max(input_train_feature), numpy.min(input_train_feature)
    
    input_train_label_file = os.path.join(input_directory, "train.label.npy")
    input_train_label = scipy.load(input_train_label_file);
    
    input_validate_feature_file = os.path.join(input_directory, "validate.feature.npy")
    input_validate_feature = scipy.load(input_validate_feature_file);
    input_validate_feature = numpy.reshape(input_validate_feature, (input_validate_feature.shape[0], 3, 32, 32))
    # print input_validate_feature.dtype, input_validate_feature.shape, numpy.max(input_validate_feature), numpy.min(input_validate_feature)
    
    input_validate_label_file = os.path.join(input_directory, "validate.label.npy")
    input_validate_label = scipy.load(input_validate_label_file);
    
    all_train_feature = numpy.concatenate((input_train_feature, input_validate_feature), axis=0);
    # print all_train_feature.dtype, all_train_feature.shape, numpy.max(all_train_feature), numpy.min(all_train_feature)
    #all_train_feature = all_train_feature.astype(numpy.float32);
    #print all_train_feature.dtype, all_train_feature.shape, numpy.max(all_train_feature), numpy.min(all_train_feature)
    
    all_train_feature *= 256
    all_train_feature = all_train_feature.astype(numpy.uint8);
    print all_train_feature.dtype, all_train_feature.shape, numpy.max(all_train_feature), numpy.min(all_train_feature)
    
    '''
    train_mean = numpy.mean(all_train_feature, axis=0)[numpy.newaxis, :];
    all_train_feature -= train_mean
    print all_train_feature.dtype, all_train_feature.shape, numpy.max(all_train_feature), numpy.min(all_train_feature)
    '''
    
    all_train_label = numpy.concatenate((input_train_label, input_validate_label), axis=0);
    # print all_train_label.dtype, all_train_label.shape
    all_train_label = all_train_label.astype(numpy.uint8);
    print all_train_label.dtype, all_train_label.shape, numpy.max(all_train_label), numpy.min(all_train_label)
    
    images_for_display = numpy.swapaxes(all_train_feature, 1, 2)
    images_for_display = numpy.swapaxes(images_for_display, 2, 3)
    matplotlib.pyplot.imshow(images_for_display[0, :, :, :])
    matplotlib.pyplot.show()
    matplotlib.pyplot.imshow(images_for_display[-1, :, :, :])
    matplotlib.pyplot.show()
    
    output_train_feature_file = os.path.join(output_directory, "train.feature.npy")
    numpy.save(output_train_feature_file, all_train_feature)

    output_train_label_file = os.path.join(output_directory, "train.label.npy")
    numpy.save(output_train_label_file, all_train_label)
    
    #
    #
    #
    #
    #
    
    input_test_feature_file = os.path.join(input_directory, "test.feature.npy")
    input_test_feature = scipy.load(input_test_feature_file);
    input_test_feature = numpy.reshape(input_test_feature, (input_test_feature.shape[0], 3, 32, 32))
    # print input_test_feature.dtype, input_test_feature.shape
    
    input_test_label_file = os.path.join(input_directory, "test.label.npy")
    input_test_label = scipy.load(input_test_label_file);
    # print input_test_label.dtype, input_test_label.shape
    
    #all_test_feature = input_test_feature.astype(numpy.float32);
    #print all_test_feature.dtype, all_test_feature.shape, numpy.max(all_test_feature), numpy.min(all_test_feature)
    
    all_test_feature = input_test_feature * 256
    all_test_feature = all_test_feature.astype(numpy.uint8);
    print all_test_feature.dtype, all_test_feature.shape, numpy.max(all_test_feature), numpy.min(all_test_feature)
    
    # all_test_feature -= train_mean
    # print all_test_feature.dtype, all_test_feature.shape, numpy.max(all_test_feature), numpy.min(all_test_feature)
    
    all_test_label = input_test_label.astype(numpy.uint8);
    print all_test_label.dtype, all_test_label.shape, numpy.max(all_test_label), numpy.min(all_test_label)
    
    images_for_display = numpy.swapaxes(all_test_feature, 1, 2)
    images_for_display = numpy.swapaxes(images_for_display, 2, 3)
    matplotlib.pyplot.imshow(images_for_display[0, :, :, :])
    matplotlib.pyplot.show()
    matplotlib.pyplot.imshow(images_for_display[-1, :, :, :])
    matplotlib.pyplot.show()
    
    output_test_feature_file = os.path.join(output_directory, "test.feature.npy")
    numpy.save(output_test_feature_file, all_test_feature)

    output_test_label_file = os.path.join(output_directory, "test.label.npy")
    numpy.save(output_test_label_file, all_test_label)
    
if __name__ == '__main__':
    input_directory = sys.argv[1]
    output_directory = sys.argv[2]
    
    cifar_parser(input_directory, output_directory)
