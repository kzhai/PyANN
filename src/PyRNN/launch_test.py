import os
import sys
#sys.path.insert(1, "/homes/kzhai/.local/lib/python2.7/site-packages")

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
                        #batch_size=0,
                        # best_model_only=False,
                        )
    # parameter set 1
    parser.add_option("--input_directory", type="string", dest="input_directory",
                      help="input directory [None]");
    parser.add_option("--model_directory", type="string", dest="model_directory",
                      help="model directory [None]");

    #parser.add_option("--batch_size", type="int", dest="batch_size",
                      #help="batch size [0]");
    # parser.add_option("--best_model_only", action="store_true", dest="best_model_only",
                      # help="best model only");

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

    test_set_x = numpy.load(os.path.join(input_directory, "test.feature.npy"))
    test_set_y = numpy.load(os.path.join(input_directory, "test.label.npy"))
    assert len(test_set_x) == len(test_set_y);

    #batch_size = options.batch_size;
    #if batch_size <= 0:
        #batch_size = test_set_x.shape[0];
    
    '''
    print "========== ========== ========== ========== =========="
    # parameter set 1
    print "model_directory=" + model_directory
    print "input_directory=" + input_directory
    print "dataset_name=" + dataset_name
    #print "batch_size=" + str(batch_size)
    print "========== ========== ========== ========== =========="
    '''
    
    if os.path.isfile(model_directory):
        model_file_path = model_directory
        prediction_loss_on_test_set, prediction_accuracy_on_test_set = evaluate_snapshot(model_file_path, test_set_x, test_set_y)
        # prediction_error_on_test_set = evaluate_snapshot(model_file_path, test_set_x, test_set_y)
        # print 'prediction accuracy is %f%% for %s' % (prediction_accuracy_on_test_set * 100., model_file_path)
        print '%f%%\t%f%%' % (prediction_accuracy_on_test_set * 100., 100 - prediction_accuracy_on_test_set * 100.)
    else:
        model_file_path = os.path.join(model_directory, "model.pkl");
        prediction_loss_on_test_set, prediction_accuracy_on_test_set = evaluate_snapshot(model_file_path, test_set_x, test_set_y)
        # print 'prediction accuracy is %f%% for %s' % (prediction_accuracy_on_test_set * 100., model_file_path)
        print '%f%%\t%f%%\t%d' % (prediction_accuracy_on_test_set * 100., 100 - prediction_accuracy_on_test_set * 100., -1)
        
        for model_file_name in os.listdir(model_directory):
            if not model_file_name.startswith("model-"):
                continue;
            # snapshot_index = int(model_file_name.split("-")[-1]);
            
            model_file_path = os.path.join(model_directory, model_file_name);
            prediction_loss_on_test_set, prediction_accuracy_on_test_set = evaluate_snapshot(model_file_path, test_set_x, test_set_y)
            # print 'prediction accuracy is %f%% for %s' % (prediction_accuracy_on_test_set * 100., model_file_path)
            
            snapshot_index = int(model_file_name.split(".")[0].split("-")[1])
            print '%f%%\t%f%%\t%d' % (prediction_accuracy_on_test_set * 100., 100 - prediction_accuracy_on_test_set * 100., snapshot_index)

def evaluate_snapshot(input_snapshot_path, test_set_x, test_set_y):
    network = cPickle.load(open(input_snapshot_path, 'rb'));

    #
    #
    #
    #
    #

    x = theano.tensor.itensor3('x')  # as many columns as context window size/lines as words in the sentence
    # m = theano.tensor.itensor3('m')  # as many columns as context window size/lines as words in the sentence
    # x = theano.tensor.imatrix('x')  # as many columns as context window size/lines as words in the sentence
    m = theano.tensor.imatrix('m')  # as many columns as context window size/lines as words in the sentence
    # y = theano.tensor.imatrix('y')  # label
    y = theano.tensor.ivector('y')  # label

    input_layers = [layer for layer in network.get_all_layers() if isinstance(layer, lasagne.layers.input.InputLayer)]
    input_layers[0].input_var = x;
    input_layers[1].input_var = m;

    # disabling dropout layers.
    test_prediction = network.get_output(deterministic=True)
    test_loss = theano.tensor.mean(theano.tensor.nnet.categorical_crossentropy(test_prediction, y))
    # As a bonus, also create an expression for the classification accuracy:
    test_accuracy = theano.tensor.mean(theano.tensor.eq(theano.tensor.argmax(test_prediction, axis=1), y), dtype=theano.config.floatX)

    validate_function = theano.function(
        inputs=[x, y, m],
        outputs=[test_loss, test_accuracy],
    )

    total_test_loss = 0;
    total_test_accuracy = 0;
    total_test_instances = 0;
    for test_sequence_x, test_sequence_y in zip(test_set_x, test_set_y):
        #context_windows = get_context_windows(test_sequence_x, window_size)
        #test_minibatch, test_minibatch_masks = get_mini_batches(context_windows, backprop_step);
        test_minibatch, test_minibatch_masks = network.get_mini_batches(test_sequence_x);
        assert len(test_minibatch) == len(test_minibatch_masks);
        assert len(test_minibatch) == len(test_sequence_x);

        minibatch_test_loss, minibatch_test_accuracy = validate_function(test_minibatch, test_sequence_y, test_minibatch_masks)

        total_test_loss += minibatch_test_loss * len(test_sequence_y)
        total_test_accuracy += minibatch_test_accuracy * len(test_sequence_y);
        total_test_instances += len(test_sequence_y);

        #test_prediction_distribution = lasagne.layers.get_output(network._neural_network, test_minibatch, test_minibatch_masks, deterministic=True).eval()

        #total_test_loss += theano.tensor.mean(theano.tensor.nnet.categorical_crossentropy(test_prediction_distribution, y)) * len(test_sequence_y)
        #total_test_accuracy += theano.tensor.mean(theano.tensor.eq(theano.tensor.argmax(test_prediction_distribution, axis=1), y), dtype=theano.config.floatX) * len(test_sequence_y);
        #total_test_instances += len(test_sequence_y);

        if total_test_instances % 1000 == 0:
            print "test progress: %d instances" % (total_test_instances)

    average_test_accuracy = total_test_accuracy / total_test_instances;
    average_test_loss = total_test_loss / total_test_instances;

    return average_test_loss, average_test_accuracy;

'''
def evaluate_snapshot_batch(input_snapshot_path, test_set_x, test_set_y):
    network = cPickle.load(open(input_snapshot_path, 'rb'));

    test_prediction_distribution = lasagne.layers.get_output(network._neural_network, test_set_x, deterministic=True).eval()
    
    # prediction_loss_on_test_set = theano.tensor.mean(theano.tensor.nnet.categorical_crossentropy(test_prediction_distribution, y))
    prediction_loss_on_test_set = 0;

    test_prediction = numpy.argmax(test_prediction_distribution, axis=1);
    test_accuracy = numpy.equal(test_prediction, test_set_y);
    prediction_accuracy_on_test_set = numpy.mean(test_accuracy);
    
    return prediction_loss_on_test_set, prediction_accuracy_on_test_set;

def evaluate_snapshot_through_graph(input_snapshot_path, test_set_x, test_set_y):
    # allocate symbolic variables for the data
    # x = theano.tensor.matrix('x')  # the data is presented as rasterized images
    x = theano.tensor.tensor4('x')  # the data is presented as rasterized images
    y = theano.tensor.ivector('y')  # the labels are presented as 1D vector of [int] labels
    
    network = cPickle.load(open(input_snapshot_path, 'rb'));
    
    # This is to establish the computational graph
    # network.get_all_layers()[0].input_var = x
    network.set_input_variable(x);
    
    # Create a train_loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the networks,
    # disabling dropout layers.
    test_prediction = network.get_output(deterministic=True)
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
'''

if __name__ == '__main__':
    launch_test()
