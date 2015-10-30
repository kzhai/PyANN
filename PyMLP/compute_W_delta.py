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
from webbrowser import Opera

def parse_args():
    parser = optparse.OptionParser()
    parser.set_defaults(# parameter set 1
                        # input_directory=None,
                        model_directory=None,
                        layer_index=1,
                        image_size="28,28",
                        tile_size="10,10"
                        )
    # parameter set 1
    # parser.add_option("--input_directory", type="string", dest="input_directory", help="input directory [None]");
    parser.add_option("--model_directory", type="string", dest="model_directory",
                      help="model directory [None]");
    parser.add_option("--layer_index", type="int", dest="layer_index",
                      help="layer index [1]");

    parser.add_option("--image_size", type="string", dest="image_size",
                      help="image size [28,28]");
    parser.add_option("--tile_size", type="string", dest="tile_size",
                      help="tile size [10,10]");

    
    (options, args) = parser.parse_args();
    return options;

def launch_test():
    """
    Demonstrate stochastic gradient descent optimization for a multilayer perceptron
    This is demonstrated on MNIST.
    """
    
    options = parse_args();

    # parameter set 1
    # assert(options.input_directory != None);
    assert(options.model_directory != None);
    
    # input_directory = options.input_directory;
    # input_directory = input_directory.rstrip("/");
    # dataset_name = os.path.basename(input_directory);
    model_directory = options.model_directory;
    layer_index = options.layer_index;

    image_size = options.image_size;
    image_size = tuple([int(x) for x in image_size.split(",")])
    tile_size = options.tile_size;
    tile_size = tuple([int(x) for x in tile_size.split(",")])
    
    print "========== ========== ========== ========== =========="
    # parameter set 1
    print "model_directory=" + model_directory
    print "layer_index=" + str(layer_index)
    
    print "image_size=%s" % str(image_size)
    print "tile_size=%s" % str(tile_size)
    print "========== ========== ========== ========== =========="
    
    compute_delta(model_directory, layer_index, image_size);

def compute_delta(model_directory, layer_index, image_size):
    W_at_epoch = {};
    for model_file_name in os.listdir(model_directory):
        if not model_file_name.startswith("model-"):
            continue;
        snapshot_index = int(model_file_name.split(".")[0].split("-")[1]);

        model_file_path = os.path.join(model_directory, model_file_name);
        network = cPickle.load(open(model_file_path, 'rb'));
        layers = lasagne.layers.get_all_layers(network._network)
        
        if layer_index < 0 or layer_index >= len(layers):
            sys.stderr.write("error: invalid layer index %d..." % layer_index);
            return;

        layer = layers[layer_index]
        
        W_at_epoch[snapshot_index] = layer.W.get_value(borrow=True).T;
        print W_at_epoch[snapshot_index].shape
    
    feature_indices_to_plot = set();
    for epoch_index in W_at_epoch:
        W_delta = numpy.abs(W_at_epoch[epoch_index] - W_at_epoch[0]);
        W_delta_sum_over_1 = numpy.mean(W_delta, axis=1);
        candidate_feature_indices = numpy.nonzero(W_delta_sum_over_1 > 0.0125)[0]
        feature_indices_to_plot.update(candidate_feature_indices)
        # W_delta_percentage = numpy.abs((W_at_epoch[epoch_index] - W_at_epoch[0]) / W_at_epoch[0]);
        # print epoch_index, numpy.max(W_delta), numpy.min(W_delta), numpy.max(W_delta_percentage), numpy.min(W_delta_percentage)
        
    print len(feature_indices_to_plot)
    
    fig, axes = matplotlib.pyplot.subplots(len(W_at_epoch), len(feature_indices_to_plot), sharex=True, sharey=True, frameon=False)
    fig.subplots_adjust(hspace=0)
    
    plot_count = 0;
    for epoch_index in sorted(W_at_epoch.keys()):
        for feature_index in feature_indices_to_plot:
            # axes[plot_count].set_frame_on(False)
            # axes[plot_count].get_xaxis().set_visible(False)
            # axes[plot_count].get_yaxis().set_visible(False)
            
            '''
            axes[plot_count].spines['right'].set_visible(False)
            axes[plot_count].spines['left'].set_visible(False)
            axes[plot_count].spines['top'].set_visible(False)
            axes[plot_count].spines['bottom'].set_visible(False)
            '''
            
            plot_count += 1;
            matplotlib.pyplot.subplot(len(W_at_epoch), len(feature_indices_to_plot), plot_count);
            matplotlib.pyplot.imshow(numpy.reshape(W_at_epoch[epoch_index][feature_index, :], image_size))
            #matplotlib.pyplot.imshow(numpy.reshape(W_at_epoch[epoch_index][feature_index, :] - W_at_epoch[0][feature_index, :], image_size))
    
    matplotlib.pyplot.show()
    
'''
fig, ax = matplotlib.pyplot.subplots()

image = numpy.random.uniform(size=(10, 10))
ax.imshow(image, cmap=matplotlib.pyplot.cm.gray, interpolation='nearest')
ax.set_title('dropped spines')

# Move left and bottom spines outward by 10 points
ax.spines['left'].set_position(('outward', 10))
ax.spines['bottom'].set_position(('outward', 10))
# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

plt.show()

fig = figure()
subplot(221)
imshow(rand(100, 100))
subplot(222)
imshow(rand(100, 100))
subplot(223)
imshow(rand(100, 100))
subplot(224)
imshow(rand(100, 100))

subplot_tool()
show()
'''
if __name__ == '__main__':
    launch_test()
