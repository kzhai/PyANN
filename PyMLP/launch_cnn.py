import os
import sys
sys.path.insert(1, "/homes/kzhai/.local/lib/python2.7/site-packages")

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

import networks
import networks.sdae

def parse_args():
    parser = optparse.OptionParser()
    parser.set_defaults(# parameter set 1
                        input_directory=None,
                        output_directory=None,
                        # pretrained_model_file=None,
                        input_shape=None,
                        
                        # parameter set 2
                        number_of_epochs=-1,
                        learning_rate=1e-3,
                        minibatch_size=100,
                        snapshot_interval=10,
                        validation_interval=1000,
                        objective_to_minimize=None,
                        
                        # parameter set 3
                        convolution_filter_numbers=None,
                        convolution_nonlinearities=None,
                        
                        # convolution_filter_sizes=None,
                        # maxpooling_sizes=None,
                        
                        # parameter set 4
                        dense_dimensions=None,
                        dense_nonlinearities=None,
                        
                        activation_parameters="1",
                        activation_styles="bernoulli",

                        # parameter set 5
                        L1_regularizer_lambdas="0",
                        L2_regularizer_lambdas="0",
                        
                        dae_regularizer_lambdas="0",
                        layer_corruption_levels="0",
                        )
    # parameter set 1
    parser.add_option("--input_directory", type="string", dest="input_directory",
                      help="input directory [None]");
    parser.add_option("--output_directory", type="string", dest="output_directory",
                      help="output directory [None]");
    parser.add_option("--input_shape", type="string", dest="input_shape",
                      help="input shape [None]");
    # parser.add_option("--pretrained_model_file", type="string", dest="pretrained_model_file",
                      # help="pretrained model file [None]");
                      
    # parameter set 2
    parser.add_option("--minibatch_size", type="int", dest="minibatch_size",
                      help="mini-batch size [100]");
    parser.add_option("--learning_rate", type="float", dest="learning_rate",
                      help="learning rate [1e-3]")
    parser.add_option("--number_of_epochs", type="int", dest="number_of_epochs",
                      help="number of epochs [-1]");
    parser.add_option("--snapshot_interval", type="int", dest="snapshot_interval",
                      help="snapshot interval in number of epochs [10]");
    parser.add_option("--validation_interval", type="int", dest="validation_interval",
                      help="validation interval in number of mini-batches [1000]");
    parser.add_option("--objective_to_minimize", type="string", dest="objective_to_minimize",
                      help="objective function to minimize [None], example, 'squared_error' represents the neural network optimizes squared error");
                      
    # parameter set 3
    parser.add_option("--convolution_filter_numbers", type="string", dest="convolution_filter_numbers",
                      help="number of convolution filters [None], example, '32,16' represents 32 and 16 filters for convolution layers respectively");
    parser.add_option("--convolution_nonlinearities", type="string", dest="convolution_nonlinearities",
                      help="activation functions of convolution layers [None], example, 'tanh,softmax' represents 2 layers with tanh and softmax activation function respectively");
                      
    '''
    parser.add_option("--convolution_filter_sizes", type="string", dest="convolution_filter_sizes",
                      help="convolution filter sizes [None], example, '5*5,6*6' represents 5*5 and 6*6 filter size for convolution layers respectively");
    parser.add_option("--maxpooling_sizes", type="string", dest="maxpooling_sizes",
                      help="max pooling sizes [None], example, '2*2,3*3' represents 2*2 and 3*3 max pooling size respectively");            
    '''
    
    # parameter set 4
    parser.add_option("--dense_dimensions", type="string", dest="dense_dimensions",
                      help="dimension of different layer [None], example, '100,500,10' represents 3 layers contains 100, 500, and 10 neurons respectively");
    parser.add_option("--dense_nonlinearities", type="string", dest="dense_nonlinearities",
                      help="activation functions of different layer [None], example, 'tanh,softmax' represents 2 layers with tanh and softmax activation function respectively");
    
    parser.add_option("--activation_parameters", type="string", dest="activation_parameters",
                      help="dropout probability of different layer [1], either one number of a list of numbers, example, '0.2' represents 0.2 dropout rate for all input+hidden layers, or '0.2,0.5' represents 0.2 dropout rate for input layer and 0.5 dropout rate for first hidden layer respectively");
    parser.add_option("--activation_styles", type="string", dest="activation_styles",
                      help="dropout style different layer [bernoulli], example, 'bernoulli,beta-bernoulli' represents 2 layers with bernoulli and beta-bernoulli dropout respectively");
                      
    # parameter set 5
    parser.add_option("--L1_regularizer_lambdas", type="string", dest="L1_regularizer_lambdas",
                      help="L1 regularization lambda [0]")
    parser.add_option("--L2_regularizer_lambdas", type="string", dest="L2_regularizer_lambdas",
                      help="L2 regularization lambda [0]")
    
    parser.add_option("--dae_regularizer_lambdas", type="string", dest="dae_regularizer_lambdas",
                      help="dae regularization lambda [0]")
    parser.add_option("--layer_corruption_levels", type="string", dest="layer_corruption_levels",
                      help="layer corruption level for pre-training [0], either one number of a list of numbers, example, '0.2' represents 0.2 corruption level for all denoising auto encoders, or '0.2,0.5' represents 0.2 corruption level for first denoising auto encoder layer and 0.5 for second one respectively");
                      
    (options, args) = parser.parse_args();
    return options;

def launch_cnn():
    """
    Demonstrate stochastic gradient descent optimization for a multilayer perceptron
    This is demonstrated on MNIST.
    """
    
    options = parse_args();

    # parameter set 2
    assert(options.minibatch_size > 0);
    minibatch_size = options.minibatch_size;
    assert options.learning_rate > 0;
    learning_rate = options.learning_rate;
    assert(options.number_of_epochs > 0);
    number_of_epochs = options.number_of_epochs;
    assert(options.validation_interval > 0);
    validation_interval = options.validation_interval;
    assert(options.snapshot_interval > 0);
    snapshot_interval = options.snapshot_interval;
    assert options.objective_to_minimize != None
    objective_to_minimize = options.objective_to_minimize;
    objective_to_minimize = getattr(lasagne.objectives, objective_to_minimize)
    
    # parameter set 3
    assert options.convolution_filter_numbers != None
    convolution_filter_numbers = [int(convolution_filter_number) for convolution_filter_number in options.convolution_filter_numbers.split(",")]
    
    assert options.convolution_nonlinearities != None
    convolution_nonlinearities = [getattr(lasagne.nonlinearities, layer_nonlinearity) for layer_nonlinearity in options.convolution_nonlinearities.split(",")]
    
    number_of_convolution_layers = len(convolution_filter_numbers);
    assert len(convolution_nonlinearities) == number_of_convolution_layers
    assert len(convolution_filter_numbers) == number_of_convolution_layers
    
    '''
    assert options.convolution_filter_sizes != None
    convolution_filter_sizes = [];
    for convolution_filter_size in options.convolution_filter_sizes.split(","):
        convolution_filter_sizes.append((int(x) for x in convolution_filter_size.split("*")))
    
    assert options.maxpooling_sizes != None
    maxpooling_sizes = [];
    for maxpooling_size in options.maxpooling_sizes.split(","):
        maxpooling_sizes.append((int(x) for x in maxpooling_size.split("*")))
    '''
    
    # parameter set 4
    assert options.dense_dimensions != None
    dense_shapes = [int(dimensionality) for dimensionality in options.dense_dimensions.split(",")]

    assert options.dense_nonlinearities != None
    dense_nonlinearities = options.dense_nonlinearities.split(",")
    dense_nonlinearities = [getattr(lasagne.nonlinearities, layer_nonlinearity) for layer_nonlinearity in dense_nonlinearities]
    
    number_of_dense_layers = len(dense_shapes);
    assert len(dense_shapes) == number_of_dense_layers;
    assert len(dense_nonlinearities) == number_of_dense_layers;

    number_of_layers = number_of_convolution_layers + number_of_dense_layers;
    
    activation_styles = options.activation_styles
    activation_styles_tokens = activation_styles.split(",")
    if len(activation_styles_tokens) == 1:
        activation_styles = [activation_styles for layer_index in xrange(number_of_layers)]
    assert len(activation_styles) == number_of_layers;
    for activation_style in activation_styles:
        assert activation_style in set(["bernoulli", "beta_bernoulli", "reciprocal_beta_bernoulli"])
    
    activation_parameters = options.activation_parameters;
    activation_parameter_tokens = activation_parameters.split(",")
    if len(activation_parameter_tokens) == 1:
        activation_parameters = [activation_parameters for layer_index in xrange(number_of_layers)]
    elif len(activation_parameter_tokens) == number_of_layers:
        activation_parameters = activation_parameter_tokens
        # [float(layer_activation_parameter) for layer_activation_parameter in activation_parameter_tokens]
    else:
        sys.stderr.write("error: unrecognized configuration for activation_parameters %s\n" % activation_parameters);
        sys.exit()
    # assert (layer_activation_parameter >= 0 for layer_activation_parameter in activation_parameters)
    # assert (layer_activation_parameter <= 1 for layer_activation_parameter in activation_parameters)
    
    for layer_index in xrange(number_of_layers):
        if activation_styles[layer_index] == "bernoulli":
            activation_parameters[layer_index] = float(activation_parameters[layer_index])
            assert activation_parameters[layer_index] <= 1;
            assert activation_parameters[layer_index] > 0;
        elif activation_styles[layer_index] == "beta_bernoulli" or activation_styles[layer_index] == "reciprocal_beta_bernoulli":
            activation_parameter_tokens = activation_parameters[layer_index].split("+");
            if len(activation_parameter_tokens) == 1:
                activation_parameters[layer_index] = (float(activation_parameter_tokens[0]), 1.0)
            elif len(activation_parameter_tokens) == 2:
                activation_parameters[layer_index] = (float(activation_parameter_tokens[0]), float(activation_parameter_tokens[1]))
            else:
                sys.stderr.write("error: unrecognized configuration for activation_style %s\n" % activation_styles[layer_index]);
                sys.exit()
            assert activation_parameters[layer_index][0] > 0;
            assert activation_parameters[layer_index][1] > 0;
    
    # parameter set 5
    L1_regularizer_lambdas = options.L1_regularizer_lambdas
    L1_regularizer_lambda_tokens = L1_regularizer_lambdas.split(",")
    if len(L1_regularizer_lambda_tokens) == 1:
        L1_regularizer_lambdas = [float(L1_regularizer_lambdas) for layer_index in xrange(number_of_layers)]
    else:
        L1_regularizer_lambdas = [float(L1_regularizer_lambda_token) for L1_regularizer_lambda_token in L1_regularizer_lambda_tokens]
    assert len(L1_regularizer_lambdas) == number_of_layers
    
    L2_regularizer_lambdas = options.L2_regularizer_lambdas
    L2_regularizer_lambda_tokens = L2_regularizer_lambdas.split(",")
    if len(L2_regularizer_lambda_tokens) == 1:
        L2_regularizer_lambdas = [float(L2_regularizer_lambdas) for layer_index in xrange(number_of_layers)]
    else:
        L2_regularizer_lambdas = [float(L2_regularizer_lambda_token) for L2_regularizer_lambda_token in L2_regularizer_lambda_tokens]
    assert len(L2_regularizer_lambdas) == number_of_layers;
    
    dae_regularizer_lambdas = options.dae_regularizer_lambdas
    dae_regularizer_lambda_tokens = dae_regularizer_lambdas.split(",")
    if len(dae_regularizer_lambda_tokens) == 1:
        dae_regularizer_lambdas = [float(dae_regularizer_lambdas) for layer_index in xrange(number_of_layers - 1)]
    else:
        dae_regularizer_lambdas = [float(dae_regularizer_lambda_token) for dae_regularizer_lambda_token in dae_regularizer_lambda_tokens]
    assert len(dae_regularizer_lambdas) == number_of_layers - 1;
    
    # parameter set 6 
    layer_corruption_levels = options.layer_corruption_levels;
    layer_corruption_level_tokens = layer_corruption_levels.split(",")
    if len(layer_corruption_level_tokens) == 1:
        layer_corruption_levels = [float(layer_corruption_levels) for layer_index in xrange(number_of_layers - 1)]
    else:
        assert len(layer_corruption_level_tokens) == number_of_layers - 1;
        layer_corruption_levels = [float(layer_corruption_level) for layer_corruption_level in layer_corruption_level_tokens]
    assert len(layer_corruption_levels) == number_of_layers - 1;
    assert (layer_corruption_level >= 0 for layer_corruption_level in layer_corruption_levels)
    assert (layer_corruption_level <= 1 for layer_corruption_level in layer_corruption_levels)
    
    # parameter set 1
    assert(options.input_shape != None);
    input_shape = [int(input_dimension) for input_dimension in options.input_shape.split(",")]
    
    assert(options.input_directory != None);
    assert(options.output_directory != None);
    
    input_directory = options.input_directory;
    input_directory = input_directory.rstrip("/");
    dataset_name = os.path.basename(input_directory);
    
    '''
    pretrained_model_file = options.pretrained_model_file;
    pretrained_model = None;
    if pretrained_model_file != None:
        assert os.path.exists(pretrained_model_file)
        pretrained_model = cPickle.load(open(pretrained_model_file, 'rb'));
    '''
    
    output_directory = options.output_directory;
    if not os.path.exists(output_directory):
        os.mkdir(output_directory);
    output_directory = os.path.join(output_directory, dataset_name);
    if not os.path.exists(output_directory):
        os.mkdir(output_directory);
    
    # create output directory
    now = datetime.datetime.now();
    suffix = now.strftime("%y%m%d-%H%M%S") + "";
    suffix += "-%s" % ("cnn");
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
    options_output_file.write("input_shape=" + str(input_shape) + "\n")
    
    # parameter set 2
    options_output_file.write("number_of_epochs=%d\n" % (number_of_epochs));
    options_output_file.write("snapshot_interval=%d\n" % (snapshot_interval))
    options_output_file.write("minibatch_size=" + str(minibatch_size) + "\n");
    options_output_file.write("learning_rate=" + str(learning_rate) + "\n");
    options_output_file.write("objective_to_minimize=%s\n" % (objective_to_minimize));
    
    # parameter set 3
    options_output_file.write("convolution_filter_numbers=%s\n" % (convolution_filter_numbers));
    options_output_file.write("convolution_nonlinearities=%s\n" % (convolution_nonlinearities));
    
    # parameter set 4
    options_output_file.write("dense_shapes=%s\n" % (dense_shapes));
    options_output_file.write("dense_nonlinearities=%s\n" % (dense_nonlinearities));
    
    options_output_file.write("activation_parameters=%s\n" % (activation_parameters));
    options_output_file.write("activation_styles=%s\n" % (activation_styles));
    
    # parameter set 5
    options_output_file.write("L1_regularizer_lambdas=%s\n" % (L1_regularizer_lambdas));
    options_output_file.write("L2_regularizer_lambdas=%s\n" % (L2_regularizer_lambdas));
    
    options_output_file.write("dae_regularizer_lambdas=%s\n" % (dae_regularizer_lambdas));
    options_output_file.write("layer_corruption_levels=%s\n" % (layer_corruption_levels));
    
    options_output_file.close()

    print "========== ========== ========== ========== =========="
    # parameter set 1
    print "output_directory=" + output_directory
    print "input_directory=" + input_directory
    print "dataset_name=" + dataset_name
    print "input_shape=" + str(input_shape)
    
    # parameter set 2
    print "number_of_epochs=%d" % (number_of_epochs);
    print "snapshot_interval=" + str(snapshot_interval);
    print "minibatch_size=" + str(minibatch_size)
    print "learning_rate=" + str(learning_rate)
    print "objective_to_minimize=%s" % (objective_to_minimize)
    
    # parameter set 3
    print "convolution_filter_numbers=%s" % (convolution_filter_numbers)
    print "convolution_nonlinearities=%s" % (convolution_nonlinearities)
    
    # parameter set 4
    print "dense_shapes=%s" % (dense_shapes)
    print "dense_nonlinearities=%s" % (dense_nonlinearities)
    
    print "activation_parameters=%s" % (activation_parameters)
    print "activation_styles=%s" % (activation_styles)
    
    # parameter set 5
    print "L1_regularizer_lambdas=%s" % (L1_regularizer_lambdas)
    print "L2_regularizer_lambdas=%s" % (L2_regularizer_lambdas);
    
    print "dae_regularizer_lambdas=%s" % (dae_regularizer_lambdas);
    print "layer_corruption_levels=%s" % (layer_corruption_levels);
    print "========== ========== ========== ========== =========="
    
    data_x = numpy.load(os.path.join(input_directory, "train.feature.npy"))
    data_y = numpy.load(os.path.join(input_directory, "train.label.npy"))
    # data_x = numpy.asarray(data_x, numpy.float32) / 256
    data_x = data_x / numpy.float32(256)
    # data_x = (data_x - numpy.float32(128)) / numpy.float32(128)
    assert data_x.shape[0] == len(data_y);
    
    input_shape.insert(0, data_x.shape[0])
    data_x = numpy.reshape(data_x, input_shape);
    
    # number_of_train = int(round(0.85 * len(data_y)));
    # number_of_train = 500000
    number_of_train = len(data_y) - 6000
    indices = range(len(data_y))
    numpy.random.shuffle(indices);
    
    train_set_x = data_x[indices[:number_of_train], :]
    train_set_y = data_y[indices[:number_of_train]]

    valid_set_x = data_x[indices[number_of_train:], :]
    valid_set_y = data_y[indices[number_of_train:]]
    
    print "successfully load data with %d for training and %d for validation..." % (train_set_x.shape[0], valid_set_x.shape[0])

    ######################
    # BUILD ACTUAL MODEL #
    ######################

    # allocate symbolic variables for the data
    # x = theano.tensor.matrix('x')  # the data is presented as rasterized images
    x = theano.tensor.tensor4('x')
    y = theano.tensor.ivector('y')  # the labels are presented as 1D vector of [int] labels
    
    input_shape[0] = None;
    
    import networks.cnn
    network = networks.cnn.ConvolutionalNeuralNetwork(
        input_data=x,
        input_shape=input_shape,
        convolution_filter_numbers=convolution_filter_numbers,
        convolution_nonlinearities=convolution_nonlinearities,
        dense_shapes=dense_shapes,
        dense_nonlinearities=dense_nonlinearities,
        activation_parameters=activation_parameters,
        activation_styles=activation_styles,
        objective_to_minimize=objective_to_minimize,
        # pretrained_model=pretrained_model
        )
    
    network.set_L1_regularizer_lambda(L1_regularizer_lambdas)
    network.set_L2_regularizer_lambda(L2_regularizer_lambdas)
    network.set_dae_regularizer_lambda(dae_regularizer_lambdas, layer_corruption_levels)
    
    ###################
    # PRE-TRAIN MODEL #
    ###################
    
    '''
    if number_of_pretrain_epochs > 0:
        network.pretrain_with_dae(data_x, layer_corruption_levels, number_of_epochs=number_of_pretrain_epochs)
    '''
    
    model_file_path = os.path.join(output_directory, 'model-%d.pkl' % (0))
    cPickle.dump(network, open(model_file_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL);
    
    # Create a train_loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy train_loss):
    train_prediction = network.get_output()
    train_loss = network.get_objective_to_minimize(y);
    # train_loss = theano.tensor.mean(lasagne.objectives.categorical_crossentropy(train_prediction, y))
    # train_loss = theano.tensor.mean(theano.tensor.nnet.categorical_crossentropy(train_prediction, y))
    train_accuracy = theano.tensor.mean(theano.tensor.eq(theano.tensor.argmax(train_prediction, axis=1), y), dtype=theano.config.floatX)
    
    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    all_mlp_params = network.get_all_params(trainable=True)
    updates = lasagne.updates.nesterov_momentum(train_loss, all_mlp_params, learning_rate, momentum=0.95)
    
    # Create a train_loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the networks,
    # disabling dropout layers.
    validate_prediction = network.get_output(deterministic=True)
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
    
    # compute number of minibatches for training, validation and testing
    # number_of_minibatches = train_set_x.get_value(borrow=True).shape[0] / minibatch_size
    number_of_minibatches = train_set_x.shape[0] / minibatch_size

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
                average_validate_loss, average_validate_accuracy = validate_function(valid_set_x, valid_set_y);
                # if we got the best validation score until now
                if average_validate_accuracy > highest_prediction_accuracy:
                    highest_prediction_accuracy = average_validate_accuracy
                    best_iteration_index = iteration_index
                    
                    # save the best model
                    print 'best model found at epoch_index %i, minibatch_index %i, average_validate_accuracy %f%%' % (epoch_index, minibatch_index + 1, average_validate_accuracy * 100)
                
                    best_model_file_path = os.path.join(output_directory, 'model.pkl')
                    cPickle.dump(network, open(best_model_file_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL);
                
                # print 'epoch_index %i, minibatch_index %i, average_validate_loss %f, average_validate_accuracy %f%%' % (epoch_index, minibatch_index, average_validate_loss, average_validate_accuracy * 100)
                    
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
    launch_cnn()