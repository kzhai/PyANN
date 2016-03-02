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
                        input_directory=None,
                        model_directory=None,
                        # batch_size=0,
                        # best_model_only=False,
                        )
    # parameter set 1
    parser.add_option("--input_directory", type="string", dest="input_directory",
                      help="input directory [None]");
    parser.add_option("--model_directory", type="string", dest="model_directory",
                      help="output directory [None]");

    # parser.add_option("--batch_size", type="int", dest="batch_size",
                      # help="batch size [0]");
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
    # test_set_x = test_set_x / numpy.float32(256)
    test_set_y = numpy.load(os.path.join(input_directory, "test.label.npy"))
    
    '''
    batch_size = options.batch_size;
    if batch_size <= 0:
        batch_size = test_set_x.shape[0];
    '''
    
    '''
    print "========== ========== ========== ========== =========="
    # parameter set 1
    print "model_directory=" + model_directory
    print "input_directory=" + input_directory
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
    
    for layer_index in xrange(1, len(generalized_dropout_layers)):
        test_layer_output = lasagne.layers.get_output(generalized_dropout_layers[layer_index], test_set_x, deterministic=True).eval();
        
        figure_path = os.path.join(model_directory, "%s-layer%d.output.pdf" % ("-".join(model_settings), layer_index))

        # Make a normed histogram. It'll be multiplied by 100 later.
        n, bins, patches = matplotlib.pyplot.hist(test_layer_output, bins=100)
        matplotlib.pyplot.rc('font', size=30)
        # matplotlib.pyplot.xlabel("dropout rate")
    
        matplotlib.pyplot.savefig(figure_path);

if __name__ == '__main__':
    launch_test()
