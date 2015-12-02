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

import layers.dropout

def parse_args():
    parser = optparse.OptionParser()
    parser.set_defaults(# parameter set 1
                        # input_directory=None,
                        model_directory=None,
                        layer_index=1,
                        image_size="28,28",
                        tile_size="32,32"
                        )
    # parameter set 1
    # parser.add_option("--input_directory", type="string", dest="input_directory",
                      # help="input directory [None]");
    parser.add_option("--model_directory", type="string", dest="model_directory",
                      help="output directory [None]");
    parser.add_option("--layer_index", type="int", dest="layer_index",
                      help="layer index [1]");
    parser.add_option("--image_size", type="string", dest="image_size",
                      help="image size [28,28]");
    parser.add_option("--tile_size", type="string", dest="tile_size",
                      help="tile size [32,32]");

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
    # print "input_directory=" + input_directory
    # print "dataset_name=" + dataset_name
    print "layer_index=" + str(layer_index)
    print "image_size=%s" % str(image_size)
    print "tile_size=%s" % str(tile_size)
    print "========== ========== ========== ========== =========="
    
    '''
    for model_file_name in os.listdir(model_directory):
        if not model_file_name.startswith("model-"):
            continue;
        snapshot_index = int(model_file_name.split(".")[0].split("-")[1]);
        
        model_file_path = os.path.join(model_directory, model_file_name);
        figure_file_path = os.path.join(model_directory, "feature-%d_figure-%d.pdf" % (layer_index, snapshot_index))
        plot_features_for_snapshot(model_file_path, layer_index - 1, image_size, tile_size, figure_file_path)
    '''
    
    model_file_path = os.path.join(model_directory, "model.pkl");
    
    figure_file_path = os.path.join(model_directory, "feature-%d.pdf" % (layer_index))
    plot_features_for_snapshot(model_file_path, layer_index - 1, image_size, tile_size, figure_file_path)
    
    figure_file_path = os.path.join(model_directory, "activation-%d.pdf" % (layer_index))
    plot_activations_for_snapshot(model_file_path, layer_index, tile_size, figure_file_path)

def plot_activations_for_snapshot(model_path, layer_index, tile_size, figure_path=None):
    network = cPickle.load(open(model_path, 'rb'));
    
    dropout_layers = lasagne.layers.get_all_layers(network.network)
    dropout_layers = [layer for layer in dropout_layers if isinstance(layer, layers.dropout.GeneralizedDropoutLayer)]
    
    if layer_index < 0 or layer_index >= len(dropout_layers):
        sys.stderr.write("error: invalid layer index %d..." % layer_index);
        return;

    layer = dropout_layers[layer_index]
    print layer.activation_probability.shape

    image = numpy.reshape(layer.activation_probability, tile_size);
    print image.shape
    
    fig, ax = matplotlib.pyplot.subplots()
    
    #ax.imshow(image, cmap=matplotlib.pyplot.cm.gray, interpolation='nearest')
    ax.imshow(image, cmap=matplotlib.pyplot.cm.gray)
    #ax.set_title('dropped spines')
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    
    if figure_path == None:
        matplotlib.pyplot.show();
    else:
        matplotlib.pyplot.savefig(figure_path);

def plot_features_for_snapshot(model_path, layer_index, image_size, tile_size, figure_path=None):
    network = cPickle.load(open(model_path, 'rb'));
    
    dense_layers = lasagne.layers.get_all_layers(network.network)
    dense_layers = [layer for layer in dense_layers if isinstance(layer, lasagne.layers.DenseLayer)]
    
    if layer_index < 0 or layer_index >= len(dense_layers):
        sys.stderr.write("error: invalid layer index %d..." % layer_index);
        return;

    layer = dense_layers[layer_index]
    print layer.W.get_value(borrow=True).shape

    # start-snippet-4
    images = tile_raster_images(
        X=layer.W.get_value(borrow=True).T,
        img_shape=image_size,
        tile_shape=tile_size,
        tile_spacing=(1, 1),
        )
    print images.shape
    
    fig, ax = matplotlib.pyplot.subplots()

    ax.imshow(images, cmap=matplotlib.pyplot.cm.gray, interpolation='nearest')
    # ax.set_title('dropped spines')
    
    # Move left and bottom spines outward by 10 points
    # ax.spines['left'].set_position(('outward', 10))
    # ax.spines['bottom'].set_position(('outward', 10))
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    # Only show ticks on the left and bottom spines
    # ax.yaxis.set_ticks_position('left')
    # ax.xaxis.set_ticks_position('bottom')
    
    if figure_path == None:
        matplotlib.pyplot.show();
    else:
        matplotlib.pyplot.savefig(figure_path);

def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    """
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.

    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)

    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats

    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not


    :returns: array suitable for viewing as an image.
    (See:`Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape    = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [
        (ishp + tsp) * tshp - tsp
        for ishp, tshp, tsp in zip(img_shape, tile_shape, tile_spacing)
    ]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image
        if output_pixel_vals:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4),
                                    dtype='uint8')
        else:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4),
                                    dtype=X.dtype)

        # colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in xrange(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = numpy.zeros(
                    out_shape,
                    dtype=dt
                ) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = numpy.zeros(out_shape, dtype=dt)

        for tile_row in xrange(tile_shape[0]):
            for tile_col in xrange(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_x = X[tile_row * tile_shape[1] + tile_col]
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(
                            this_x.reshape(img_shape))
                    else:
                        this_img = this_x.reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                        tile_row * (H + Hs): tile_row * (H + Hs) + H,
                        tile_col * (W + Ws): tile_col * (W + Ws) + W
                    ] = this_img * c
        return out_array

def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar
  
if __name__ == '__main__':
    launch_test()
    
