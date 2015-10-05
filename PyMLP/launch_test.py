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

import lasagne

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

    print "========== ========== ========== ========== =========="
    # parameter set 1
    print "model_directory=" + model_directory
    print "input_directory=" + input_directory
    print "dataset_name=" + dataset_name
    # print "dictionary file=" + str(dict_file)
    print "========== ========== ========== ========== =========="
    
    test_set_x = numpy.load(os.path.join(input_directory, "test.feature.npy"))
    test_set_y = numpy.load(os.path.join(input_directory, "test.label.npy"))
    
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... testing the model'
    
    for model_file_name in os.listdir(model_directory):
        if not model_file_name.startswith("model-"):
            continue;
        #snapshot_index = int(model_file_name.split("-")[-1]);
        
        model_file_path = os.path.join(model_directory, model_file_name);
        prediction_loss_on_test_set, prediction_accuracy_on_test_set = plot_snapshot(model_file_path, test_set_x, test_set_y)
        print 'prediction accuracy is %f%% for %s' % (prediction_accuracy_on_test_set * 100., model_file_path)
    
    model_file_path = os.path.join(model_directory, "best_model.pkl");
    prediction_loss_on_test_set, prediction_accuracy_on_test_set = plot_snapshot(model_file_path, test_set_x, test_set_y)
    #prediction_error_on_test_set = plot_snapshot(model_file_path, test_set_x, test_set_y)
    print 'prediction accuracy is %f%% for %s' % (prediction_accuracy_on_test_set * 100., model_file_path)

def plot_snapshot(input_snapshot_path, test_set_x, test_set_y):
    # allocate symbolic variables for the data
    x = theano.tensor.matrix('x')  # the data is presented as rasterized images
    y = theano.tensor.ivector('y')  # the labels are presented as 1D vector of [int] labels
    
    network = cPickle.load(open(input_snapshot_path, 'rb'));
    
    # This is to establish the computational graph
    lasagne.layers.get_all_layers(network)[0].input_var = x
    
    # Create a train_loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = theano.tensor.mean(theano.tensor.nnet.categorical_crossentropy(test_prediction, y))
    # As a bonus, also create an expression for the classification accuracy:
    test_accuracy = theano.tensor.mean(theano.tensor.eq(theano.tensor.argmax(test_prediction, axis=1), y), dtype=theano.config.floatX)
    
    # compiling a Theano function that computes the mistakes that are made by the model on a minibatch
    test_function = theano.function(
        inputs=[x, y],
        outputs=[test_loss, test_accuracy],
    )
    
    # test it on the test set
    prediction_loss_on_test_set, prediction_accuracy_on_test_set = test_function(test_set_x, test_set_y);
    
    return prediction_loss_on_test_set, prediction_accuracy_on_test_set;
    
if __name__ == '__main__':
    launch_test()
    