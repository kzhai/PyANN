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

import matplotlib.pyplot

from layers.dropout import sample_activation_probability

def parse_args():
    parser = optparse.OptionParser()
    parser.set_defaults(# parameter set 1
                        # input_directory=None,
                        feature_model="bernoulli",
                        feature_model_parameter="1",
                        feature_dimension=1000,
                        feature_model_file=None,
                        )
    # parameter set 1
    # parser.add_option("--input_directory", type="string", dest="input_directory",
                      # help="input directory [None]");
    parser.add_option("--feature_model", type="string", dest="feature_model",
                      help="feature model [bernoulli, beta-bernoulli, reciprocal_beta_bernoulli]");
    parser.add_option("--feature_model_parameter", type="string", dest="feature_model_parameter",
                      help="model alpha [1]");
    parser.add_option("--feature_dimension", type="int", dest="feature_dimension",
                      help="feature dimension [1000]");
    # parser.add_option("--image_size", type="string", dest="image_size",
                      # help="image size [28,28]");
    # parser.add_option("--tile_size", type="string", dest="tile_size",
                      # help="tile size [10,10]");
                      
    parser.add_option("--feature_model_file", type="string", dest="feature_model_file",
                      help="feature model file [None]");

    (options, args) = parser.parse_args();
    return options;

def plot_feature_model():
    """
    Demonstrate stochastic gradient descent optimization for a multilayer perceptron
    This is demonstrated on MNIST.
    """
    
    options = parse_args();

    # parameter set 1
    # assert(options.input_directory != None);
    assert(options.feature_model != None);
    
    # input_directory = options.input_directory;
    # input_directory = input_directory.rstrip("/");
    # dataset_name = os.path.basename(input_directory);
    feature_model = options.feature_model;
    feature_model_parameter = options.feature_model_parameter;
    
    if feature_model == "bernoulli":
        feature_model_parameter = float(feature_model_parameter)
        assert feature_model_parameter <= 1;
        assert feature_model_parameter > 0;
    elif feature_model == "beta_bernoulli" or feature_model == "reciprocal_beta_bernoulli" or feature_model == "reverse_reciprocal_beta_bernoulli" or feature_model == "mixed_beta_bernoulli":
        feature_model_parameter_tokens = feature_model_parameter.split("+");
        if len(feature_model_parameter_tokens) == 1:
            feature_model_parameter = (float(feature_model_parameter_tokens[0]), 1.0)
        elif len(feature_model_parameter_tokens) == 2:
            feature_model_parameter = (float(feature_model_parameter_tokens[0]), float(feature_model_parameter_tokens[1]))
        else:
            sys.stderr.write("error: unrecognized configuration for activation_style %s\n" % feature_model);
            sys.exit()
        assert feature_model_parameter[0] > 0;
        assert feature_model_parameter[1] > 0;
    elif feature_model == "reciprocal":
        feature_model_parameter = float(feature_model_parameter)
        assert feature_model_parameter > 0;
        
    feature_dimension = options.feature_dimension;
    figure_path = options.feature_model_file
    '''
    image_size = options.image_size;
    image_size = tuple([int(x) for x in image_size.split(",")])
    tile_size = options.tile_size;
    tile_size = tuple([int(x) for x in tile_size.split(",")])
    '''
    
    print "========== ========== ========== ========== =========="
    # parameter set 1
    print "feature_model=" + feature_model
    # print "input_directory=" + input_directory
    # print "dataset_name=" + dataset_name
    print "feature_model_parameter=" + str(feature_model_parameter)
    # print "image_size=%s" % str(image_size)
    # print "tile_size=%s" % str(tile_size)
    print "========== ========== ========== ========== =========="
    
    activation_probability = sample_activation_probability(feature_dimension, feature_model, feature_model_parameter)
    plot_activation_probability_hist(activation_probability, 100, figure_path)
    #plot_activation_probability_barh(activation_probability)

def plot_activation_probability_barh(activation_probability, figure_path=None):
    # Make a normed histogram. It'll be multiplied by 100 later.
    # n, bins, patches = matplotlib.pyplot.hist(activation_probability, bins=100, normed=1)
    #n, bins, patches = matplotlib.pyplot.hist(activation_probability)
    matplotlib.pyplot.barh(numpy.arange(len(activation_probability)), activation_probability, align='center', alpha=0.5)
    
    if figure_path == None:
        matplotlib.pyplot.show();
    else:
        matplotlib.pyplot.savefig(figure_path);

def plot_activation_probability_hist(activation_probability, bins, figure_path=None):
    # Make a normed histogram. It'll be multiplied by 100 later.
    #n, bins, patches = matplotlib.pyplot.hist(activation_probability, bins=bins, normed=1)
    n, bins, patches = matplotlib.pyplot.hist(activation_probability, bins=bins)
    matplotlib.pyplot.xlim((0, 1))
    matplotlib.pyplot.rc('font', size=30)
    #matplotlib.pyplot.xlabel("dropout rate")

    if figure_path == None:
        matplotlib.pyplot.show();
    else:
        matplotlib.pyplot.savefig(figure_path);

if __name__ == '__main__':
    plot_feature_model()    
