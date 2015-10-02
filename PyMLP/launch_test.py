import os
import sys
import timeit

import cPickle
import numpy
import scipy

import theano
import theano.tensor

import datetime
import optparse

import multilayer_perceptron
import neural_network_layer
import launch_train

def parse_args():
    parser = optparse.OptionParser()
    parser.set_defaults(# parameter set 1
                        input_directory=None,
                        model_directory=None,
                        )
    # parameter set 1
    parser.add_option("--input_directory", type="string", dest="input_directory",
                      help="input directory [None]");
    parser.add_option("--model_directory", type="string", dest="model_directory",
                      help="output directory [None]");

    (options, args) = parser.parse_args();
    return options;

def launch_test():
    """
    Demonstrate stochastic gradient descent optimization for a multilayer perceptron
    This is demonstrated on MNIST.
    """
    
    options = parse_args();

    # parameter set 1
    assert(options.input_directory != None);
    assert(options.model_directory != None);
    
    input_directory = options.input_directory;
    input_directory = input_directory.rstrip("/");
    dataset_name = os.path.basename(input_directory);
    model_directory = options.model_directory;

    # store all the options to a file
    options_output_file = open(model_directory + "option.txt", 'w');
    # parameter set 1
    options_output_file.write("input_directory=" + input_directory + "\n");
    options_output_file.write("dataset_name=" + dataset_name + "\n");
    options_output_file.write("model_directory=" + model_directory + "\n");
    # options_output_file.write("vocabulary_path=" + str(dict_file) + "\n");
    options_output_file.close()

    print "========== ========== ========== ========== =========="
    # parameter set 1
    print "model_directory=" + model_directory
    print "input_directory=" + input_directory
    print "dataset_name=" + dataset_name
    # print "dictionary file=" + str(dict_file)
    print "========== ========== ========== ========== =========="
    
    data_x = numpy.load(os.path.join(input_directory, "test.feature.npy"))
    data_y = numpy.load(os.path.join(input_directory, "test.label.npy"))
    test_set_x, test_set_y = launch_train.shared_dataset(data_x, data_y)
    
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... testing the model'
    
    for model_file_name in os.listdir(model_directory):
        if not model_file_name.startswith("model-"):
            continue;
        #snapshot_index = int(model_file_name.split("-")[-1]);
        
        model_file_path = os.path.join(model_directory, model_file_name);
        prediction_error_on_test_set = evaluate_snapshot(model_file_path, test_set_x, test_set_y)
        print 'prediction error is %f%% for %s' % (prediction_error_on_test_set * 100., model_file_path)
    
    model_file_path = os.path.join(model_directory, "best_model.pkl");
    prediction_error_on_test_set = evaluate_snapshot(model_file_path, test_set_x, test_set_y)
    #prediction_error_on_test_set = evaluate_snapshot(model_file_path, test_set_x, test_set_y)
    print 'prediction error is %f%% for %s' % (prediction_error_on_test_set * 100., model_file_path)

def evaluate_snapshot(input_snapshot_path, test_set_x, test_set_y):
    # allocate symbolic variables for the data
    x = theano.tensor.matrix('x')  # the data is presented as rasterized images
    y = theano.tensor.ivector('y')  # the labels are presented as 1D vector of [int] labels
    
    classifier = multilayer_perceptron.MultiLayerPerceptron.load_model(x, input_snapshot_path);
    
    # compiling a Theano function that computes the mistakes that are made by the model on a minibatch
    test_model = theano.function(
            inputs=[],
            outputs=[theano.tensor.mean(theano.tensor.eq(theano.tensor.argmax(classifier._output, axis=1), y))],
            givens={
                x: test_set_x,
                y: test_set_y
            }
        )
    
    # test it on the test set
    prediction_error_on_test_set = test_model()
    
    return prediction_error_on_test_set;
    
if __name__ == '__main__':
    launch_test()
    