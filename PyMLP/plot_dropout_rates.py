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
                        model_directory=None,
                        layer_index=1,
                        )
    # parameter set 1
    #parser.add_option("--input_directory", type="string", dest="input_directory",
                      #help="input directory [None]");
    parser.add_option("--model_directory", type="string", dest="model_directory",
                      help="output directory [None]");
    parser.add_option("--layer_index", type="int", dest="layer_index",
                      help="layer index [1]");
    #parser.add_option("--image_size", type="string", dest="image_size",
                      #help="image size [28,28]");
    #parser.add_option("--tile_size", type="string", dest="tile_size",
                      #help="tile size [10,10]");

    (options, args) = parser.parse_args();
    return options;

def plot_dropout_rates():
    """
    Demonstrate stochastic gradient descent optimization for a multilayer perceptron
    This is demonstrated on MNIST.
    """
    
    options = parse_args();

    # parameter set 1
    #assert(options.input_directory != None);
    assert(options.model_directory != None);
    
    #input_directory = options.input_directory;
    #input_directory = input_directory.rstrip("/");
    #dataset_name = os.path.basename(input_directory);
    model_directory = options.model_directory;
    layer_index = options.layer_index;
    '''
    image_size = options.image_size;
    image_size = tuple([int(x) for x in image_size.split(",")])
    tile_size = options.tile_size;
    tile_size = tuple([int(x) for x in tile_size.split(",")])
    '''
    
    print "========== ========== ========== ========== =========="
    # parameter set 1
    print "model_directory=" + model_directory
    #print "input_directory=" + input_directory
    #print "dataset_name=" + dataset_name
    print "layer_index=" + str(layer_index)
    #print "image_size=%s" % str(image_size)
    #print "tile_size=%s" % str(tile_size)
    print "========== ========== ========== ========== =========="
    
    '''
    for model_file_name in os.listdir(model_directory):
        if not model_file_name.startswith("model-"):
            continue;
        snapshot_index = int(model_file_name.split(".")[0].split("-")[1]);
        
        model_file_path = os.path.join(model_directory, model_file_name);
        figure_file_path = os.path.join(model_directory, "layer-%d_figure-%d.pdf" % (layer_index, snapshot_index))
        plot_snapshot(model_file_path, layer_index, figure_file_path)
    '''
    
    model_file_path = os.path.join(model_directory, "best_model.pkl");
    figure_file_path = os.path.join(model_directory, "layer-%d_dropout-rates.pdf" % (layer_index))
    plot_snapshot(model_file_path, layer_index, figure_file_path)

def plot_snapshot(model_path, layer_index, figure_path=None):
    network = cPickle.load(open(model_path, 'rb'));
    
    layers = lasagne.layers.get_all_layers(network._network)
    
    dropout_layers = [layer for layer in layers if isinstance(layer, GeneralizedDropoutLayer)];
    
    activation_probability = dropout_layers[layer_index-1].activation_probability;
    
    # Make a normed histogram. It'll be multiplied by 100 later.
    n, bins, patches = matplotlib.pyplot.hist(activation_probability, bins=100, normed=1)

    if figure_path == None:
        matplotlib.pyplot.show();
    else:
        matplotlib.pyplot.savefig(figure_path);

if __name__ == '__main__':
    plot_dropout_rates()
    