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

def build_mlp(
        input=None,
        layer_nonlinearities=None,
        layer_shapes=None,
        layer_dropout_rates=None
        ):
    # print layer_nonlinearities
    #layer_nonlinearities = [getattr(lasagne.nonlinearities, layer_nonlinearity) for layer_nonlinearity in layer_nonlinearities]
    
    network = lasagne.layers.InputLayer(shape=(None, layer_shapes[0]), input_var=input)
    #print network.shape, network.output_shape
    for layer_index in xrange(1, len(layer_shapes)):
        if layer_dropout_rates is not None and layer_dropout_rates[layer_index - 1] > 0:
            network = lasagne.layers.DropoutLayer(network, p=layer_dropout_rates[layer_index - 1])
            #print network.input_shape, network.output_shape
        
        layer_shape = layer_shapes[layer_index]
        layer_nonlinearity = layer_nonlinearities[layer_index - 1];
        network = lasagne.layers.DenseLayer(network, layer_shape, nonlinearity=layer_nonlinearity)
        #print network.input_shape, network.output_shape
        
    return network;

def parse_args():
    parser = optparse.OptionParser()
    parser.set_defaults(# parameter set 1
                        input_directory=None,
                        output_directory=None,
                        
                        # parameter set 2
                        number_of_epochs=-1,
                        minibatch_size=100,
                        snapshot_interval=10,
                        validation_interval=1000,
                        
                        # parameter set 3
                        learning_rate=1e-3,
                        
                        # parameter set 4
                        layer_dimensions=None,
                        layer_nonlinearities=None,
                        layer_dropout_rates=None,
                        
                        # parameter set 5
                        L1_regularizer_lambdas="0",
                        L2_regularizer_lambdas="0",
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
    
    # parameter set 4
    parser.add_option("--layer_dimensions", type="string", dest="layer_dimensions",
                      help="dimension of different layer [None], example, '100,500,10' represents 3 layers contains 100, 500, and 10 neurons respectively");
    parser.add_option("--layer_nonlinearities", type="string", dest="layer_nonlinearities",
                      help="activation functions of different layer [None], example, 'tanh,softmax' represents 2 layers with tanh and softmax activation function respectively");                      
    parser.add_option("--layer_dropout_rates", type="string", dest="layer_dropout_rates",
                      help="dropout probability of different layer [None], either one number of a list of numbers, example, '0.2' represents 0.2 dropout rate for all input+hidden layers, or '0.2,0.5' represents 0.2 dropout rate for input layer and 0.5 dropout rate for first hidden layer respectively");  

    # parameter set 5
    parser.add_option("--L1_regularizer_lambdas", type="string", dest="L1_regularizer_lambdas",
                      help="L1 regularization lambda [0]")
    parser.add_option("--L2_regularizer_lambdas", type="string", dest="L2_regularizer_lambdas",
                      help="L2 regularization lambda [0]")

    (options, args) = parser.parse_args();
    return options;

def launch_train():
    """
    Demonstrate stochastic gradient descent optimization for a multilayer perceptron
    This is demonstrated on MNIST.
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
    
    # parameter set 4
    assert options.layer_dimensions != None
    layer_shapes = [int(dimensionality) for dimensionality in options.layer_dimensions.split(",")]
    number_of_layers = len(layer_shapes) - 1;

    assert options.layer_nonlinearities != None
    layer_nonlinearities = options.layer_nonlinearities.split(",")
    layer_nonlinearities = [getattr(lasagne.nonlinearities, layer_nonlinearity) for layer_nonlinearity in layer_nonlinearities]
    assert len(layer_nonlinearities) == number_of_layers;
    
    layer_dropout_rates = options.layer_dropout_rates;
    if layer_dropout_rates is not None:
        layer_dropout_rate_tokens = layer_dropout_rates.split(",")
        if len(layer_dropout_rate_tokens) == 1:
            layer_dropout_rates = [float(layer_dropout_rates) for layer_index in xrange(number_of_layers)]
        else:
            assert len(layer_dropout_rate_tokens) == number_of_layers;
            layer_dropout_rates = [float(layer_dropout_rate) for layer_dropout_rate in layer_dropout_rate_tokens]
            
        assert (layer_dropout_rate >= 0 for layer_dropout_rate in layer_dropout_rates)
        assert (layer_dropout_rate <= 1 for layer_dropout_rate in layer_dropout_rates)
        
    # parameter set 5
    L1_regularizer_lambdas = options.L1_regularizer_lambdas
    L1_regularizer_lambda_tokens = L1_regularizer_lambdas.split(",")
    if len(L1_regularizer_lambda_tokens) == 1:
        L1_regularizer_lambdas = [float(L1_regularizer_lambdas) for layer_index in xrange(number_of_layers)]
    else:
        assert len(L1_regularizer_lambda_tokens) == number_of_layers;
        L1_regularizer_lambdas = [float(L1_regularizer_lambda_token) for L1_regularizer_lambda_token in L1_regularizer_lambda_tokens]
    
    L2_regularizer_lambdas = options.L2_regularizer_lambdas
    L2_regularizer_lambda_tokens = L2_regularizer_lambdas.split(",")
    if len(L2_regularizer_lambda_tokens) == 1:
        L2_regularizer_lambdas = [float(L2_regularizer_lambdas) for layer_index in xrange(number_of_layers)]
    else:
        assert len(L2_regularizer_lambda_tokens) == number_of_layers;
        L2_regularizer_lambdas = [float(L2_regularizer_lambda_token) for L2_regularizer_lambda_token in L2_regularizer_lambda_tokens]
        
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
    
    # create output directory
    now = datetime.datetime.now();
    suffix = now.strftime("%y%m%d-%H%M%S") + "";
    suffix += "-%s" % ("mlp");
    suffix += "-E%d" % (number_of_epochs);
    suffix += "-S%d" % (snapshot_interval);
    suffix += "-B%d" % (minibatch_size);
    suffix += "-aa%f" % (learning_rate);
    # suffix += "-l1r%f" % (L1_regularizer_lambdas);
    # suffix += "-l2r%d" % (L2_regularizer_lambdas);
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

    # parameter set 4
    options_output_file.write("layer_shapes=%s\n" % (layer_shapes));
    options_output_file.write("layer_nonlinearities=%s\n" % (layer_nonlinearities));
    options_output_file.write("layer_dropout_rates=%s\n" % (layer_dropout_rates));
    # parameter set 5
    options_output_file.write("L1_regularizer_lambdas=%s\n" % (L1_regularizer_lambdas));
    options_output_file.write("L2_regularizer_lambdas=%s\n" % (L2_regularizer_lambdas));
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
    # parameter set 4
    print "layer_shapes=%s" % (layer_shapes)
    print "layer_nonlinearities=%s" % (layer_nonlinearities)
    print "layer_dropout_rates=%s" % (layer_dropout_rates)
    # parameter set 5
    print "L1_regularizer_lambdas=%s" % (L1_regularizer_lambdas)
    print "L2_regularizer_lambdas=%s" % (L2_regularizer_lambdas);
    print "========== ========== ========== ========== =========="
    
    data_x = numpy.load(os.path.join(input_directory, "train.feature.npy"))
    data_y = numpy.load(os.path.join(input_directory, "train.label.npy"))
    #data_x = numpy.asarray(data_x, numpy.float32) / 256
    data_x = data_x / numpy.float32(256)
    #print data_x.dtype
    
    assert data_x.shape[0] == len(data_y);
    number_of_train = int(round(0.8 * len(data_y)));
    indices = range(len(data_y))
    numpy.random.shuffle(indices);
    
    train_set_x = data_x[indices[:number_of_train], :]
    train_set_y = data_y[indices[:number_of_train]]
    
    valid_set_x = data_x[indices[number_of_train:], :]
    valid_set_y = data_y[indices[number_of_train:]]
    
    print "successfully load data with %d for training and %d for validation..." % (train_set_x.shape[0], valid_set_x.shape[0])
    
    # compute number of minibatches for training, validation and testing
    # number_of_minibatches = train_set_x.get_value(borrow=True).shape[0] / minibatch_size
    number_of_minibatches = train_set_x.shape[0] / minibatch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################

    # allocate symbolic variables for the data
    x = theano.tensor.matrix('inputs')  # the data is presented as rasterized images
    y = theano.tensor.ivector('outputs')  # the labels are presented as 1D vector of [int] labels

    network = build_mlp(x,
                        layer_nonlinearities,  # = [theano.tensor.nnet.sigmoid, theano.tensor.nnet.softmax],
                        layer_shapes,
                        layer_dropout_rates
                        )
    
    # Create a train_loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy train_loss):
    train_prediction = lasagne.layers.get_output(network)
    # train_loss = theano.tensor.mean(lasagne.objectives.categorical_crossentropy(train_prediction, y))
    train_loss = theano.tensor.mean(theano.tensor.nnet.categorical_crossentropy(train_prediction, y))
    train_accuracy = theano.tensor.mean(theano.tensor.eq(theano.tensor.argmax(train_prediction, axis=1), y), dtype=theano.config.floatX)
    
    # We could add some weight decay as well here, see lasagne.regularization.
    network_layers = lasagne.layers.get_all_layers(network);
    L1_regularizer_layer_lambdas = {temp_layer:L1_regularizer_lambda for temp_layer, L1_regularizer_lambda in zip(network_layers[1:], L1_regularizer_lambdas)};
    L1_regularizer = lasagne.regularization.regularize_layer_params_weighted(L1_regularizer_layer_lambdas, lasagne.regularization.l1)
    L2_regularizer_layer_lambdas = {temp_layer:L2_regularizer_lambda for temp_layer, L2_regularizer_lambda in zip(network_layers[1:], L2_regularizer_lambdas)};
    L2_regularizer = lasagne.regularization.regularize_layer_params_weighted(L2_regularizer_layer_lambdas, lasagne.regularization.l2)
    train_loss += L1_regularizer + L2_regularizer

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    all_network_params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(train_loss, all_network_params, learning_rate, momentum=0.9)

    # Create a train_loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    validate_prediction = lasagne.layers.get_output(network, deterministic=True)
    validate_loss = theano.tensor.mean(theano.tensor.nnet.categorical_crossentropy(validate_prediction, y))
    # As a bonus, also create an expression for the classification accuracy:
    validate_accuracy = theano.tensor.mean(theano.tensor.eq(theano.tensor.argmax(validate_prediction, axis=1), y), dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training train_loss:
    train_function = theano.function(
        inputs=[x, y],
        outputs=[train_loss, train_accuracy],
        updates=updates
    )

    # Compile a second function computing the validation train_loss and accuracy:
    validate_function = theano.function(
        inputs=[x, y],
        outputs=[validate_loss, validate_accuracy],
    )

    ###############
    # TRAIN MODEL #
    ###############
    
    highest_prediction_accuracy = 0
    best_iteration_index = 0
    start_time = timeit.default_timer()

    # Finally, launch the training loop.
    # We iterate over epochs:
    for epoch_index in range(number_of_epochs):
        # In each epoch_index, we do a full pass over the training data:
        clock_epoch = time.time();
        for minibatch_index in xrange(number_of_minibatches):
            iteration_index = epoch_index * number_of_minibatches + minibatch_index
            
            minibatch_x = train_set_x[minibatch_index * minibatch_size:(minibatch_index + 1) * minibatch_size, :]
            minibatch_y = train_set_y[minibatch_index * minibatch_size:(minibatch_index + 1) * minibatch_size]
            average_train_loss, average_train_accuracy = train_function(minibatch_x, minibatch_y)

            # And a full pass over the validation data:
            if (iteration_index + 1) % validation_interval == 0:
                prediction_loss_on_validation_set, prediction_accuracy_on_validation_set = validate_function(valid_set_x, valid_set_y);
                # if we got the best validation score until now
                if prediction_accuracy_on_validation_set > highest_prediction_accuracy:
                    highest_prediction_accuracy = prediction_accuracy_on_validation_set
                    best_iteration_index = iteration_index
                    
                    # save the best model
                    print 'best model found at epoch_index %i, minibatch_index %i, prediction_accuracy_on_validation_set %f%%' % (epoch_index, minibatch_index, prediction_accuracy_on_validation_set * 100)
                
                    best_model_file_path = os.path.join(output_directory, 'best_model.pkl')
                    cPickle.dump(network, open(best_model_file_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL);

        clock_epoch = time.time() - clock_epoch;
    
        print 'epoch_index %i, average_train_loss %f, average_train_accuracy %f%%, running time %fs' % (epoch_index, average_train_loss, average_train_accuracy * 100, clock_epoch)
        
        if (epoch_index + 1) % snapshot_interval == 0:
            model_file_path = os.path.join(output_directory, 'model-%d.pkl' % (epoch_index + 1))
            cPickle.dump(network, open(model_file_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL);
        
    end_time = timeit.default_timer()
    print "Optimization complete..."
    print "Best validation score of %f%% obtained at iteration %i" % (highest_prediction_accuracy * 100., best_iteration_index);
    print >> sys.stderr, ('The code for file ' + 
                          os.path.split(__file__)[1] + 
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

if __name__ == '__main__':
    launch_train()
