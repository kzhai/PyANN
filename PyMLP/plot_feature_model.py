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

from layers.dropout import GeneralizedDropoutLayer

def parse_args():
    parser = optparse.OptionParser()
    parser.set_defaults(# parameter set 1
                        #input_directory=None,
                        feature_model="bernoulli",
                        feature_model_parameter=1,
                        feature_dimension=1000,
                        )
    # parameter set 1
    #parser.add_option("--input_directory", type="string", dest="input_directory",
                      #help="input directory [None]");
    parser.add_option("--feature_model", type="string", dest="feature_model",
                      help="output directory [None]");
    parser.add_option("--feature_model_parameter", type="float", dest="feature_model_parameter",
                      help="model alpha [1]");
    parser.add_option("--feature_dimension", type="int", dest="feature_dimension",
                      help="feature dimension [1000]");
    #parser.add_option("--image_size", type="string", dest="image_size",
                      #help="image size [28,28]");
    #parser.add_option("--tile_size", type="string", dest="tile_size",
                      #help="tile size [10,10]");

    (options, args) = parser.parse_args();
    return options;

def plot_feature_model():
    """
    Demonstrate stochastic gradient descent optimization for a multilayer perceptron
    This is demonstrated on MNIST.
    """
    
    options = parse_args();

    # parameter set 1
    #assert(options.input_directory != None);
    assert(options.feature_model != None);
    
    #input_directory = options.input_directory;
    #input_directory = input_directory.rstrip("/");
    #dataset_name = os.path.basename(input_directory);
    feature_model = options.feature_model;
    feature_model_parameter = options.feature_model_parameter;
    feature_dimension = options.feature_dimension;
    '''
    image_size = options.image_size;
    image_size = tuple([int(x) for x in image_size.split(",")])
    tile_size = options.tile_size;
    tile_size = tuple([int(x) for x in tile_size.split(",")])
    '''
    
    print "========== ========== ========== ========== =========="
    # parameter set 1
    print "feature_model=" + feature_model
    #print "input_directory=" + input_directory
    #print "dataset_name=" + dataset_name
    print "feature_model_parameter=" + str(feature_model_parameter)
    #print "image_size=%s" % str(image_size)
    #print "tile_size=%s" % str(tile_size)
    print "========== ========== ========== ========== =========="
    
    '''
    for model_file_name in os.listdir(feature_model):
        if not model_file_name.startswith("model-"):
            continue;
        snapshot_index = int(model_file_name.split(".")[0].split("-")[1]);
        
        model_file_path = os.path.join(feature_model, model_file_name);
        figure_file_path = os.path.join(feature_model, "layer-%d_figure-%d.pdf" % (feature_model_parameter, snapshot_index))
        plot_feature_mode(model_file_path, feature_model_parameter, figure_file_path)
    
    model_file_path = os.path.join(feature_model, "best_model.pkl");
    figure_file_path = os.path.join(feature_model, "layer-%d_dropout-rates.pdf" % (feature_model_parameter))
    '''
    
    if feature_model == "bernoulli":
        activation_probability = numpy.zeros(feature_dimension) + feature_model_parameter;
    elif feature_model == "beta-bernoulli":        
        shape_alpha = feature_model_parameter / numpy.arange(1, feature_dimension + 1);
        shape_beta = 1.0;
        
        activation_probability = numpy.zeros(feature_dimension);
        for index in xrange(feature_dimension):
            activation_probability[index] = numpy.random.beta(shape_alpha[index], shape_beta);
    elif feature_model == "reciprocal":
        activation_probability = feature_model_parameter / numpy.arange(1, feature_dimension + 1);
        activation_probability = numpy.clip(activation_probability, 0., 1.);
    else:
        sys.stderr.write("erro: unrecognized configuration...\n");
        sys.exit();

    plot_feature_mode(feature_model_parameter, feature_dimension)

def plot_feature_mode(activation_probability, bins=100, figure_path=None):
    # Make a normed histogram. It'll be multiplied by 100 later.
    #n, bins, patches = matplotlib.pyplot.hist(activation_probability, bins=100, normed=1)
    n, bins, patches = matplotlib.pyplot.hist(activation_probability, bins=bins)

    if figure_path == None:
        matplotlib.pyplot.show();
    else:
        matplotlib.pyplot.savefig(figure_path);

if __name__ == '__main__':
    plot_feature_model()
    