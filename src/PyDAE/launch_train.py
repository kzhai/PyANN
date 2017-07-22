import os
import sys
import timeit

import cPickle
import numpy
import scipy

import theano
import theano.tensor

import time
import datetime
import optparse

import lasagne

#from scripts import tile_raster_images

try:
    import PIL.Image as Image
except ImportError:
    import Image

def parse_args():
    parser = optparse.OptionParser()
    parser.set_defaults(# parameter set 1
                        input_directory=None,
                        output_directory=None,
                        
                        # parameter set 2
                        number_of_epochs=-1,
                        minibatch_size=1,
                        snapshot_interval=10,
                        validation_interval=1000,
                        
                        # parameter set 3
                        learning_rate=1e-3,
                        #learning_rate_decay=0,
                        objective_to_minimize="None",

                        # parameter set 4
                        layer_dimension=-1,
                        layer_nonlinearity=None,
                        layer_corruption_level=0,
                        
                        # parameter set 5
                        L1_regularizer_lambda=0,
                        L2_regularizer_lambda=0,
                        )
    # parameter set 1
    parser.add_option("--input_directory", type="string", dest="input_directory",
                      help="input directory [None]");
    parser.add_option("--output_directory", type="string", dest="output_directory",
                      help="output directory [None]");
                      
    # parameter set 2
    parser.add_option("--minibatch_size", type="int", dest="minibatch_size",
                      help="mini-batch size [100]");
    parser.add_option("--number_of_epochs", type="int", dest="number_of_epochs",
                      help="number of epochs [-1]");
    parser.add_option("--snapshot_interval", type="int", dest="snapshot_interval",
                      help="snapshot interval in number of epochs [10]");
    parser.add_option("--validation_interval", type="int", dest="validation_interval",
                      help="validation interval in number of mini-batches [1000]");
    # parser.add_option("--improvement_threshold", type="float", dest="improvement_threshold",
                      # help="improvement threshold [0.01]")
    
    # parameter set 3
    parser.add_option("--learning_rate", type="float", dest="learning_rate",
                      help="learning rate [1e-3]")
    #parser.add_option("--learning_rate_decay", type="float", dest="learning_rate_decay",
                      #help="learning rate [0 - no learning rate decay]")
    parser.add_option("--objective_to_minimize", type="string", dest="objective_to_minimize",
                      help="objective function to minimize [None], example, 'squared_error' represents the neural network optimizes squared error");    
    
    # parameter set 4
    parser.add_option("--layer_dimension", type="int", dest="layer_dimension",
                      help="dimension of auto-encoder layer [-1]");
    parser.add_option("--layer_nonlinearity", type="string", dest="layer_nonlinearity",
                      help="activation functions of auto-encoder layer [None]");                      
    parser.add_option("--layer_corruption_level", type="float", dest="layer_corruption_level",
                      help="corruption level of auto-encoder layer [0]");  

    # parameter set 5
    parser.add_option("--L1_regularizer_lambda", type="float", dest="L1_regularizer_lambda",
                      help="L1 regularization lambda [0]")
    parser.add_option("--L2_regularizer_lambda", type="float", dest="L2_regularizer_lambda",
                      help="L2 regularization lambda [0]")

    (options, args) = parser.parse_args();
    return options;

def launch_train():
    """
    """
    
    options = parse_args();

    # parameter set 2
    assert(options.minibatch_size > 0);
    minibatch_size = options.minibatch_size;
    assert(options.number_of_epochs > 0);
    number_of_epochs = options.number_of_epochs;
    assert(options.validation_interval > 0);
    validation_interval = options.validation_interval;
    assert(options.snapshot_interval > 0);
    snapshot_interval = options.snapshot_interval;
    
    # parameter set 3
    assert options.learning_rate > 0;
    learning_rate = options.learning_rate;
    objective_to_minimize = getattr(lasagne.objectives, options.objective_to_minimize)
    
    # parameter set 4
    assert options.layer_dimension > 0;
    layer_dimension = options.layer_dimension

    assert options.layer_nonlinearity != None
    layer_nonlinearity = getattr(lasagne.nonlinearities, options.layer_nonlinearity)
    
    layer_corruption_level = options.layer_corruption_level;
    assert (layer_corruption_level >= 0 and layer_corruption_level <= 1);
        
    # parameter set 5
    L1_regularizer_lambda = options.L1_regularizer_lambda
    L1_regularizer_lambdas = [L1_regularizer_lambda]
    
    L2_regularizer_lambda = options.L2_regularizer_lambda
    L2_regularizer_lambdas = [L2_regularizer_lambda]
    
    # parameter set 1
    assert(options.input_directory != None);
    assert(options.output_directory != None);
    
    input_directory = options.input_directory;
    input_directory = input_directory.rstrip("/");
    dataset_name = os.path.basename(input_directory);
    
    output_directory = options.output_directory;
    if not os.path.exists(output_directory):
        os.mkdir(output_directory);
    output_directory = os.path.join(output_directory, dataset_name);
    if not os.path.exists(output_directory):
        os.mkdir(output_directory);
    
    #
    #
    #
    #
    #
    
    data_x = numpy.load(os.path.join(input_directory, "train.feature.npy"))
    data_y = numpy.load(os.path.join(input_directory, "train.label.npy"))
    
    assert data_x.shape[0] == len(data_y);
    
    input_shape = list(data_x.shape[1:]);
    input_shape.insert(0, None)
    
    indices = range(len(data_y))
    numpy.random.shuffle(indices);
    
    train_set_x = data_x[indices, :]
    train_set_y = data_y[indices]

    print "successfully load data with %d for training..." % (train_set_x.shape[0])
    
    #
    #
    #
    #
    #
    
    # create output directory
    now = datetime.datetime.now();
    suffix = now.strftime("%y%m%d-%H%M%S") + "";
    suffix += "-%s" % ("dae");
    suffix += "-E%d" % (number_of_epochs);
    suffix += "-S%d" % (snapshot_interval);
    suffix += "-B%d" % (minibatch_size);
    suffix += "-aa%f" % (learning_rate);
    # suffix += "-l1r%f" % (L1_regularizer_lambda);
    # suffix += "-l2r%d" % (L2_regularizer_lambda);
    suffix += "/";
    
    output_directory = os.path.join(output_directory, suffix);
    os.mkdir(os.path.abspath(output_directory));
    
    # store all the options to a file
    options_output_file = open(output_directory + "option.txt", 'w');
    # parameter set 1
    options_output_file.write("input_directory=" + input_directory + "\n");
    options_output_file.write("dataset_name=" + dataset_name + "\n");
    # options_output_file.write("vocabulary_path=" + str(dict_file) + "\n");
    # parameter set 2
    options_output_file.write("number_of_epochs=%d\n" % (number_of_epochs));
    options_output_file.write("minibatch_size=" + str(minibatch_size) + "\n");
    # parameter set 3
    options_output_file.write("learning_rate=" + str(learning_rate) + "\n");
    options_output_file.write("objective_to_minimize=" + str(objective_to_minimize) + "\n");
    # parameter set 4
    options_output_file.write("layer_dimension=%s\n" % (layer_dimension));
    options_output_file.write("layer_nonlinearity=%s\n" % (layer_nonlinearity));
    options_output_file.write("layer_corruption_level=%s\n" % (layer_corruption_level));
    # parameter set 5
    options_output_file.write("L1_regularizer_lambda=%s\n" % (L1_regularizer_lambda));
    options_output_file.write("L2_regularizer_lambda=%s\n" % (L2_regularizer_lambda));
    
    options_output_file.close()

    print "========== ========== ========== ========== =========="
    # parameter set 1
    print "output_directory=" + output_directory
    print "input_directory=" + input_directory
    print "dataset_name=" + dataset_name
    # print "dictionary file=" + str(dict_file)
    # parameter set 2
    print "number_of_epochs=%d" % (number_of_epochs);
    print "snapshot_interval=" + str(snapshot_interval);
    print "minibatch_size=" + str(minibatch_size)
    # parameter set 3
    print "learning_rate=" + str(learning_rate)
    print "objective_to_minimize=" + str(objective_to_minimize)
    # parameter set 4
    print "layer_dimension=%s" % (layer_dimension)
    print "layer_nonlinearity=%s" % (layer_nonlinearity)
    print "layer_corruption_level=%s" % (layer_corruption_level)
    # parameter set 5
    print "L1_regularizer_lambda=%s" % (L1_regularizer_lambda)
    print "L2_regularizer_lambda=%s" % (L2_regularizer_lambda);
    print "========== ========== ========== ========== =========="
    
    ######################
    # BUILD ACTUAL MODEL #
    ######################

    # allocate symbolic variables for the data
    x = theano.tensor.matrix('inputs')  # the data is presented as rasterized images
    # y = theano.tensor.ivector('outputs')  # the labels are presented as 1D vector of [int] labels

    #
    #
    #
    #
    #
    
    input_layer = lasagne.layers.InputLayer(shape=input_shape, input_var=x);
    
    import dae
    network = dae.DenoisingAutoEncoder(
            input_network=input_layer,
            layer_dimension=layer_dimension,
            encoder_nonlinearity=layer_nonlinearity,
            decoder_nonlinearity=layer_nonlinearity,
            objective_to_minimize=objective_to_minimize,
            corruption_level=layer_corruption_level)
    
    network.set_L1_regularizer_lambda(L1_regularizer_lambdas)
    network.set_L2_regularizer_lambda(L2_regularizer_lambdas)
    
    #
    #
    #
    #
    #
    
    # Create a train_loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy train_loss):
    train_prediction = network.get_output()
    train_loss = network.get_objective_to_minimize();
    # train_accuracy = theano.tensor.mean(theano.tensor.eq(theano.tensor.argmax(train_prediction, axis=1), y), dtype=theano.config.floatX)
    
    # theano.printing.debugprint(train_loss)
    
    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    all_params = network.get_network_params(trainable=True)
    updates = lasagne.updates.nesterov_momentum(train_loss, all_params, learning_rate, momentum=0.95)
    # updates = lasagne.updates.sgd(train_loss, all_params, learning_rate)

    '''
    # Create a train_loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the networks,
    # disabling dropout layers.
    validate_prediction = lasagne.layers.get_output(networks, deterministic=True)
    validate_loss = theano.tensor.mean(theano.tensor.nnet.categorical_crossentropy(validate_prediction, y))
    # As a bonus, also create an expression for the classification accuracy:
    validate_accuracy = theano.tensor.mean(theano.tensor.eq(theano.tensor.argmax(validate_prediction, axis=1), y), dtype=theano.config.floatX)
    '''

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training train_loss:
    train_function = theano.function(
        inputs=[x],
        outputs=[train_loss, train_prediction],
        updates=updates
    )

    '''
    # Compile a second function computing the validation train_loss and accuracy:
    validate_function = theano.function(
        inputs=[x, y],
        outputs=[validate_loss, validate_accuracy],
    )
    '''
    
    ###############
    # TRAIN MODEL #
    ###############
    start_train = timeit.default_timer()
    
    model_file_path = os.path.join(output_directory, 'model-%d.pkl' % (0))
    cPickle.dump(network, open(model_file_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL);

    # compute number of minibatches for training, validation and testing
    # number_of_minibatches = train_set_x.get_value(borrow=True).shape[0] / minibatch_size
    number_of_minibatches = train_set_x.shape[0] / minibatch_size

    # Finally, launch the training loop.
    # We iterate over epochs:
    for epoch_index in range(number_of_epochs):
        # In each epoch_index, we do a full pass over the training data:
        start_epoch = timeit.default_timer()
        minibatch_train_losses = []
        for minibatch_index in xrange(number_of_minibatches):

            model_file_path = os.path.join(output_directory, 'model-%d.pkl' % (epoch_index + 1))
            cPickle.dump(network, open(model_file_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL);
            
            iteration_index = epoch_index * number_of_minibatches + minibatch_index
            
            minibatch_x = train_set_x[minibatch_index * minibatch_size:(minibatch_index + 1) * minibatch_size, :]
            # minibatch_y = train_set_y[minibatch_index * minibatch_size:(minibatch_index + 1) * minibatch_size]
            minibatch_train_loss, average_train_prediction = train_function(minibatch_x)
            minibatch_train_losses.append(minibatch_train_loss);

        end_epoch = timeit.default_timer()
        print 'epoch %i, average train loss %f, running time %fs' % (epoch_index, numpy.mean(minibatch_train_losses), (end_epoch - start_epoch))

        if (epoch_index + 1) % snapshot_interval == 0:
            model_file_path = os.path.join(output_directory, 'model-%d.pkl' % (epoch_index + 1))
            cPickle.dump(network, open(model_file_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL);
            
    model_file_path = os.path.join(output_directory, 'model-%d.pkl' % (epoch_index + 1))
    cPickle.dump(network, open(model_file_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL);
    
    end_train = timeit.default_timer()
    print "Optimization complete..."
    print >> sys.stderr, ('The code for file ' + 
                          os.path.split(__file__)[1] + 
                          ' ran for %.2fm' % ((end_train - start_train) / 60.))
    
    '''
    image = Image.fromarray(
        tile_raster_images(X=network.network.W_encoder.get_value(borrow=True).T,
                           img_shape=(28, 28), tile_shape=(10, 10),
                           tile_spacing=(1, 1)))
    image.save(os.path.join(output_directory, 'filters.png'))
    '''
    
if __name__ == '__main__':
    launch_train()
