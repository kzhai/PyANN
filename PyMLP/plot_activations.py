import os
import sys
sys.path.insert(1, "/homes/kzhai/.local/lib/python2.7/site-packages")

import timeit

import cPickle
import numpy
import matplotlib
import matplotlib.pyplot
import scipy

import theano
import theano.tensor

import datetime
import optparse

import lasagne
import layers

def parse_args():
    parser = optparse.OptionParser()
    parser.set_defaults(# parameter set 1
                        input_file=None,
                        model_directory=None,
                        batch_size=0,
                        # best_model_only=False,
                        )
    # parameter set 1
    parser.add_option("--input_file", type="string", dest="input_file",
                      help="input directory [None]");
    parser.add_option("--model_directory", type="string", dest="model_directory",
                      help="output directory [None]");

    parser.add_option("--batch_size", type="int", dest="batch_size",
                      help="batch size [0]");
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
    
    input_file = options.input_directory;
    input_file = input_file.rstrip("/");
    dataset_name = os.path.basename(input_file);
    model_directory = options.model_directory;

    test_set_x = numpy.load(os.path.join(input_file, "test.feature.npy"))
    # test_set_x = test_set_x / numpy.float32(256)
    test_set_y = numpy.load(os.path.join(input_file, "test.label.npy"))
    
    batch_size = options.batch_size;
    if batch_size <= 0:
        batch_size = test_set_x.shape[0];
    
    '''
    print "========== ========== ========== ========== =========="
    # parameter set 1
    print "model_directory=" + model_directory
    print "input_file=" + input_file
    print "dataset_name=" + dataset_name
    print "batch_size=" + str(batch_size)
    print "========== ========== ========== ========== =========="
    '''
    
    model_directory = model_directory.rstrip("/");
    model_settings = os.path.basename(model_directory);
    model_settings = model_settings.split("-")[:2];
    
    model_file_path = os.path.join(model_directory, "model.pkl");
    network = cPickle.load(open(model_file_path, 'rb'));
    
    dense_layers = [layer for layer in network.get_all_layers() if isinstance(layer, lasagne.layers.DenseLayer)];
    # dropout_layers = [layer for layer in network.get_all_layers() if isinstance(layer, lasagne.layers.DropoutLayer)];
    generalized_dropout_layers = [layer for layer in network.get_all_layers() if isinstance(layer, layers.dropout.GeneralizedDropoutLayer)];
    assert len(generalized_dropout_layers) == len(dense_layers)
    
    indices = range(test_set_x.shape[0])
    numpy.random.shuffle(indices);
    test_subset_x = test_set_x[indices[:batch_size], :];
    
    for layer_index in xrange(1, len(generalized_dropout_layers)):
        test_layer_output = lasagne.layers.get_output(generalized_dropout_layers[layer_index], test_subset_x, deterministic=True).eval();

        figure_path = os.path.join(model_directory, "%s-layer%d-output.pdf" % ("-".join(model_settings), layer_index))
        
        test_layer_output = numpy.reshape(test_layer_output, test_layer_output.shape[0] * test_layer_output.shape[1])
        
        min_value = numpy.min(test_layer_output);
        max_value = numpy.max(test_layer_output);
        
        # Make a normed histogram. It'll be multiplied by 100 later.
        n, bins, patches = matplotlib.pyplot.hist(test_layer_output, bins=100)
        matplotlib.pyplot.xlim((min_value, max_value))
        matplotlib.pyplot.rc('font', size=25)
        # matplotlib.pyplot.xlabel("dropout rate")
    
        matplotlib.pyplot.savefig(figure_path);

if __name__ == '__main__':
    launch_test()
