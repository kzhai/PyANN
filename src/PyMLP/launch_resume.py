import os
import sys
import shutil
import re

import cPickle
import numpy
import scipy

import theano
import theano.tensor

import timeit
import datetime
import optparse

import lasagne

def parse_args():
    parser = optparse.OptionParser()
    parser.set_defaults(# parameter set 1
                        input_directory=None,
                        #output_directory=None,
                        model_directory=None,
                        model_snapeshot=None,
                        
                        # parameter set 2
                        number_of_epochs=-1,
                        minibatch_size=100,
                        snapshot_interval=-1,
                        validation_interval=1000,

                        # parameter set 3
                        learning_rate=1e-2,
                        learning_rate_decay=0,
                        #objective_to_minimize=None,

                        # parameter set 4
                        #layer_dimensions=None,
                        #layer_nonlinearities=None,

                        #layer_activation_parameters="1",
                        #layer_activation_styles="bernoulli",

                        # parameter set 5
                        #L1_regularizer_lambdas="0",
                        #L2_regularizer_lambdas="0",
                        
                        #dae_regularizer_lambdas="0",
                        #layer_corruption_levels="0",
                        
                        # parameter set 6
                        #number_of_training_data=-1,
                        #snapshot_index=-1,
                        )
    # parameter set 1
    parser.add_option("--input_directory", type="string", dest="input_directory",
                      help="input directory [None]");
    #parser.add_option("--output_directory", type="string", dest="output_directory",
                      #help="output directory [None]");
    parser.add_option("--model_directory", type="string", dest="model_directory",
                      help="model directory [None]");
    parser.add_option("--model_snapshot", type="string", dest="model_snapshot",
                      help="model snapshot [None]");
                      
    # parameter set 2
    parser.add_option("--minibatch_size", type="int", dest="minibatch_size",
                      help="mini-batch size [100]");
    parser.add_option("--number_of_epochs", type="int", dest="number_of_epochs",
                      help="number of epochs [-1]");
    parser.add_option("--snapshot_interval", type="int", dest="snapshot_interval",
                      help="snapshot interval in number of epochs [-1]");
    parser.add_option("--validation_interval", type="int", dest="validation_interval",
                      help="validation interval in number of mini-batches [1000]");
    # parser.add_option("--improvement_threshold", type="float", dest="improvement_threshold",
                      # help="improvement threshold [0.01]")
    
    # parameter set 3
    parser.add_option("--learning_rate", type="float", dest="learning_rate",
                      help="learning rate [1e-3]")
    parser.add_option("--learning_rate_decay", type="float", dest="learning_rate_decay",
                      help="learning rate decay [0 - no decay]")
    parser.add_option("--objective_to_minimize", type="string", dest="objective_to_minimize",
                      help="objective function to minimize [None], example, 'squared_error' represents the neural network optimizes squared error");

    '''
    # parameter set 4
    parser.add_option("--layer_dimensions", type="string", dest="layer_dimensions",
                      help="dimension of different layer [None], example, '100,500,10' represents 3 layers contains 100, 500, and 10 neurons respectively");
    parser.add_option("--layer_nonlinearities", type="string", dest="layer_nonlinearities",
                      help="activation functions of different layer [None], example, 'tanh,softmax' represents 2 layers with tanh and softmax activation function respectively");
                    
    parser.add_option("--layer_activation_parameters", type="string", dest="layer_activation_parameters",
                      help="dropout probability of different layer [1], either one number of a list of numbers, example, '0.2' represents 0.2 dropout rate for all input+hidden layers, or '0.2,0.5' represents 0.2 dropout rate for input layer and 0.5 dropout rate for first hidden layer respectively");
    parser.add_option("--layer_activation_styles", type="string", dest="layer_activation_styles",
                      help="dropout style different layer [bernoulli], example, 'bernoulli,beta-bernoulli' represents 2 layers with bernoulli and beta-bernoulli dropout respectively");
    # parser.add_option("--layer_latent_feature_alphas", type="string", dest="layer_latent_feature_alphas",
                      # help="alpha for latent feature ");
                                            
    # parameter set 5
    parser.add_option("--L1_regularizer_lambdas", type="string", dest="L1_regularizer_lambdas",
                      help="L1 regularization lambda [0]")
    parser.add_option("--L2_regularizer_lambdas", type="string", dest="L2_regularizer_lambdas",
                      help="L2 regularization lambda [0]")
    
    parser.add_option("--dae_regularizer_lambdas", type="string", dest="dae_regularizer_lambdas",
                      help="dae regularization lambda [0]")
    parser.add_option("--layer_corruption_levels", type="string", dest="layer_corruption_levels",
                      help="layer corruption level for pre-training [0], either one number of a list of numbers, example, '0.2' represents 0.2 corruption level for all denoising auto encoders, or '0.2,0.5' represents 0.2 corruption level for first denoising auto encoder layer and 0.5 for second one respectively");

    # parameter set 6
    parser.add_option("--number_of_training_data", type="int", dest="number_of_training_data",
                      help="training data size [-1]");
    # parser.add_option("--number_of_pretrain_epochs", type="int", dest="number_of_pretrain_epochs",
                      # help="number of pretrain epochs [0 - no pre-training]");
    '''

    #parser.add_option("--snapshot_index", type="int", dest="snapshot_index",
                      #help="snapshot index [-1 - use the best model]");

    (options, args) = parser.parse_args();
    return options;

def launch_resume():
    """
    Demonstrate stochastic gradient descent optimization for a multilayer perceptron
    """

    options = parse_args();

    # parameter set 2
    assert(options.minibatch_size > 0);
    minibatch_size = options.minibatch_size;
    assert(options.number_of_epochs > 0);
    number_of_epochs = options.number_of_epochs;
    assert(options.validation_interval > 0);
    validation_interval = options.validation_interval;
    #assert(options.snapshot_interval > 0);
    snapshot_interval = options.snapshot_interval;

    # parameter set 3
    assert options.learning_rate > 0;
    initial_learning_rate = options.learning_rate;
    assert options.learning_rate_decay >= 0;
    learning_rate_decay = options.learning_rate_decay;

    #assert options.objective_to_minimize != None
    #objective_to_minimize = options.objective_to_minimize;
    #objective_to_minimize = getattr(lasagne.objectives, objective_to_minimize)

    '''
    # parameter set 4
    assert options.layer_dimensions != None
    layer_dimensions = [int(dimensionality) for dimensionality in options.layer_dimensions.split(",")]
    number_of_layers = len(layer_dimensions);

    assert options.layer_nonlinearities != None
    layer_nonlinearities = options.layer_nonlinearities.split(",")
    layer_nonlinearities = [getattr(lasagne.nonlinearities, layer_nonlinearity) for layer_nonlinearity in layer_nonlinearities]
    assert len(layer_nonlinearities) == number_of_layers;

    layer_activation_styles = options.layer_activation_styles;
    layer_activation_style_tokens = layer_activation_styles.split(",")
    if len(layer_activation_style_tokens) == 1:
        layer_activation_styles = [layer_activation_styles for layer_index in xrange(number_of_layers)]
    elif len(layer_activation_style_tokens) == number_of_layers:
        layer_activation_styles = layer_activation_style_tokens
        # [float(layer_activation_parameter) for layer_activation_parameter in layer_activation_parameter_tokens]
    else:
        sys.stderr.write("error: unrecognized configuration for layer_activation_styles %s\n" % layer_activation_styles);
        sys.exit()
        
    assert len(layer_activation_styles) == number_of_layers;
    for layer_activation_style in layer_activation_styles:
        assert layer_activation_style in set(["bernoulli", "beta_bernoulli", "reciprocal_beta_bernoulli", "reverse_reciprocal_beta_bernoulli", "mixed_beta_bernoulli"])
    
    layer_activation_parameters = options.layer_activation_parameters;
    # if layer_activation_parameters is not None:
    layer_activation_parameter_tokens = layer_activation_parameters.split(",")
    if len(layer_activation_parameter_tokens) == 1:
        layer_activation_parameters = [layer_activation_parameters for layer_index in xrange(number_of_layers)]
    elif len(layer_activation_parameter_tokens) == number_of_layers:
        layer_activation_parameters = layer_activation_parameter_tokens
        # [float(layer_activation_parameter) for layer_activation_parameter in layer_activation_parameter_tokens]
    else:
        sys.stderr.write("error: unrecognized configuration for layer_activation_parameters %s\n" % layer_activation_parameters);
        sys.exit()
    # assert (layer_activation_parameter >= 0 for layer_activation_parameter in layer_activation_parameters)
    # assert (layer_activation_parameter <= 1 for layer_activation_parameter in layer_activation_parameters)
    
    for layer_index in xrange(number_of_layers):
        if layer_activation_styles[layer_index] == "bernoulli":
            layer_activation_parameters[layer_index] = float(layer_activation_parameters[layer_index])
            assert layer_activation_parameters[layer_index] <= 1;
            assert layer_activation_parameters[layer_index] > 0;
        elif layer_activation_styles[layer_index] == "beta_bernoulli" or layer_activation_styles[layer_index] == "reciprocal_beta_bernoulli" or layer_activation_styles[layer_index] == "reverse_reciprocal_beta_bernoulli" or layer_activation_styles[layer_index] == "mixed_beta_bernoulli":
            layer_activation_parameter_tokens = layer_activation_parameters[layer_index].split("+");
            if len(layer_activation_parameter_tokens) == 1:
                layer_activation_parameters[layer_index] = (float(layer_activation_parameter_tokens[0]), 1.0)
            elif len(layer_activation_parameter_tokens) == 2:
                layer_activation_parameters[layer_index] = (float(layer_activation_parameter_tokens[0]), float(layer_activation_parameter_tokens[1]))
            else:
                sys.stderr.write("error: unrecognized configuration for layer_activation_style %s\n" % layer_activation_styles[layer_index]);
                sys.exit()
            assert layer_activation_parameters[layer_index][0] > 0;
            assert layer_activation_parameters[layer_index][1] > 0;
            
            if layer_activation_styles[layer_index] == "mixed_beta_bernoulli":
                assert layer_activation_parameters[layer_index][0] < 1;
    '''

    '''
    layer_latent_feature_alphas = options.layer_latent_feature_alphas;
    if layer_latent_feature_alphas is not None:
        layer_latent_feature_alpha_tokens = layer_latent_feature_alphas.split(",")
        if len(layer_latent_feature_alpha_tokens) == 1:
            layer_latent_feature_alphas = [float(layer_latent_feature_alphas) for layer_index in xrange(number_of_layers)]
        else:
            assert len(layer_latent_feature_alpha_tokens) == number_of_layers;
            layer_latent_feature_alphas = [float(layer_latent_feature_alpha) for layer_latent_feature_alpha in layer_latent_feature_alpha_tokens]
    else:
        layer_latent_feature_alphas = [0 for layer_index in xrange(number_of_layers)]
    assert (layer_latent_feature_alpha >= 0 for layer_latent_feature_alpha in layer_latent_feature_alphas)
    '''

    '''
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
    '''

    # parameter set 1
    assert (options.input_directory != None);
    assert (options.model_directory != None);
    #assert (options.output_directory != None);
    
    input_directory = options.input_directory;
    #input_directory = input_directory.rstrip("/");
    #dataset_name = os.path.basename(input_directory);

    output_directory = options.model_directory;
    if not os.path.exists(output_directory):
        sys.stderr.write("model directory %s not exists...\n" % (output_directory));
        return;

    data_x = numpy.load(os.path.join(input_directory, "train.feature.npy"))
    data_y = numpy.load(os.path.join(input_directory, "train.label.npy"))
    assert data_x.shape[0] == len(data_y);

    train_indices = numpy.load(os.path.join(output_directory, "train.index.npy"));
    valid_indices = numpy.load(os.path.join(output_directory, "valid.index.npy"));

    train_set_x = data_x[train_indices, :]
    train_set_y = data_y[train_indices]

    valid_set_x = data_x[valid_indices, :]
    valid_set_y = data_y[valid_indices]

    print "successfully load data with %d for training and %d for validation..." % (train_set_x.shape[0], valid_set_x.shape[0])

    test_set_x = numpy.load(os.path.join(input_directory, "test.feature.npy"))
    test_set_y = numpy.load(os.path.join(input_directory, "test.label.npy"))
    assert test_set_x.shape[0] == len(test_set_y);

    print "successfully load data with %d for testing..." % (test_set_x.shape[0])

    # load the existing model
    if options.model_snapshot==None:
        model_snapshot = os.path.join(output_directory, "model.pkl");
    else:
        model_snapshot = options.model_snapshot;

    if not os.path.exists(model_snapshot):
        sys.stderr.write("error: model file %s not found...\n" % (model_snapshot));
        return;

    #number_of_layers = len(network.get_all_layers()) - 1;

    '''
    output_file = options.output_directory;
    if not os.path.exists(output_file):
        os.mkdir(output_file);
    output_file = os.path.join(output_file, dataset_name);
    if not os.path.exists(output_file):
        os.mkdir(output_file);

    pretrained_model_file = options.pretrained_model_file;
    pretrained_model = None;
    if pretrained_model_file != None:
        assert os.path.exists(pretrained_model_file)
        pretrained_model = cPickle.load(open(pretrained_model_file, 'rb'));
    
    output_directory = options.output_directory;
    if not os.path.exists(output_directory):
        os.mkdir(output_directory);
    output_directory = os.path.join(output_directory, dataset_name);
    if not os.path.exists(output_directory):
        os.mkdir(output_directory);
    '''

    #
    #
    #
    #
    #
    
    # create output directory
    now = datetime.datetime.now();
    suffix = now.strftime("%y%m%d-%H%M%S") + "";
    '''
    suffix += "-%s" % ("mlp");
    suffix += "-T%d" % (number_of_training_data);
    suffix += "-E%d" % (number_of_epochs);
    #suffix += "-S%d" % (snapshot_interval);
    suffix += "-B%d" % (minibatch_size);
    suffix += "-lr%f" % (initial_learning_rate);
    suffix += "-ld%f" % (initial_learning_decay);
    # suffix += "-l1r%f" % (L1_regularizer_lambdas);
    # suffix += "-l2r%d" % (L2_regularizer_lambdas);
    suffix += "/";
    
    output_directory = os.path.join(output_directory, suffix);
    os.mkdir(os.path.abspath(output_directory));
    '''
    #
    #
    #
    #
    #

    print "========== ========== ========== ========== =========="
    # parameter set 1
    #print "output_directory=" + output_directory
    print "input_directory=" + input_directory
    print "output_directory=" + output_directory
    #print "dataset_name=" + dataset_name
    print "model_snapshot=%s" % model_snapshot
    # print "dictionary file=" + str(dict_file)

    # parameter set 2
    print "number_of_epochs=%d" % (number_of_epochs);
    print "minibatch_size=" + str(minibatch_size)
    print "snapshot_interval=" + str(snapshot_interval);
    print "validation_interval=%d" % validation_interval;
    
    # parameter set 3
    print "learning_rate=" + str(initial_learning_rate)
    print "learning_rate_decay=" + str(learning_rate_decay)
    #print "objective_to_minimize=%s" % (objective_to_minimize)

    '''
    # parameter set 4
    print "layer_dimensions=%s" % (layer_dimensions)
    print "layer_nonlinearities=%s" % (layer_nonlinearities)
    
    print "layer_activation_parameters=%s" % (layer_activation_parameters)
    print "layer_activation_styles=%s" % (layer_activation_styles)
    
    # parameter set 5
    print "L1_regularizer_lambdas=%s" % (L1_regularizer_lambdas)
    print "L2_regularizer_lambdas=%s" % (L2_regularizer_lambdas);
    
    print "dae_regularizer_lambdas=%s" % (dae_regularizer_lambdas);
    print "layer_corruption_levels=%s" % (layer_corruption_levels);
    
    # paramter set 6
    print "number_of_training_data=%d" % (number_of_training_data);
    '''
    print "========== ========== ========== ========== =========="

    #
    #
    #
    #
    #

    cPickle.dump(options, open(os.path.join(output_directory, "option.pkl"), 'wb'),
                 protocol=cPickle.HIGHEST_PROTOCOL);

    #
    #
    #
    #
    #

    ######################
    # BUILD ACTUAL MODEL #
    ######################

    # allocate symbolic variables for the data
    x = theano.tensor.matrix('x')  # the data is presented as rasterized images
    y = theano.tensor.ivector('y')  # the labels are presented as 1D vector of [int] labels
    lr = theano.tensor.scalar('learning_rate');
    
    #input_layer = lasagne.layers.InputLayer(shape=input_shape, input_var=x)

    '''
    import mlp
    network = mlp.MultiLayerPerceptron(
        input_network=input_layer,
        layer_dimensions=layer_dimensions,
        layer_nonlinearities=layer_nonlinearities,
        layer_activation_parameters=layer_activation_parameters,
        layer_activation_styles=layer_activation_styles,
        objective_to_minimize=objective_to_minimize,
        # pretrained_model=pretrained_model
        )
    '''

    network = cPickle.load(open(model_snapshot, 'rb'));

    # This is to establish the computational graph
    # network.get_all_layers()[0].input_var = x
    network.set_input_variable(x);

    #network.set_L1_regularizer_lambda(L1_regularizer_lambdas)
    #network.set_L2_regularizer_lambda(L2_regularizer_lambdas)
    #network.set_dae_regularizer_lambda(dae_regularizer_lambdas, layer_corruption_levels)
    
    ########################
    # BUILD LOSS FUNCTIONS #
    ########################
    
    # Create a train_loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy train_loss):
    train_prediction = network.get_output()
    train_loss = network.get_objective_to_minimize(y);
    # train_loss = theano.tensor.mean(lasagne.objectives.categorical_crossentropy(train_prediction, y))
    train_accuracy = theano.tensor.mean(theano.tensor.eq(theano.tensor.argmax(train_prediction, axis=1), y), dtype=theano.config.floatX)
    #train_accuracy = network.get_objective_to_minimize(y, objective_to_minimize="categorical_accuracy");

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    all_params = network.get_network_params(trainable=True)
    updates = lasagne.updates.nesterov_momentum(train_loss, all_params, lr, momentum=0.95)
    
    # Create a train_loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the networks,
    # disabling dropout layers.
    validate_prediction = network.get_output(deterministic=True)
    validate_loss = network.get_objective_to_minimize(y, deterministic=True);
    #validate_loss = theano.tensor.mean(theano.tensor.nnet.categorical_crossentropy(validate_prediction, y), dtype=theano.config.floatX)
    # As a bonus, also create an expression for the classification accuracy:
    validate_accuracy = theano.tensor.mean(theano.tensor.eq(theano.tensor.argmax(validate_prediction, axis=1), y), dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training train_loss:
    train_function = theano.function(
        inputs=[x, y, lr],
        outputs=[train_loss, train_accuracy],
        updates=updates
    )
    
    # Compile a second function computing the validation train_loss and accuracy:
    validate_function = theano.function(
        inputs=[x, y],
        outputs=[validate_loss, validate_accuracy],
    )
    
    ########################
    # START MODEL TRAINING #
    ########################

    highest_average_validate_accuracy = 0
    best_iteration_index = 0

    start_train = timeit.default_timer()

    # model_file_path = os.path.join(output_directory, 'model-%d.pkl' % (0))
    # cPickle.dump(network, open(model_file_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL);

    # compute number of minibatches for training, validation and testing
    # number_of_minibatches = train_set_x.get_value(borrow=True).shape[0] / minibatch_size
    number_of_minibatches = train_set_x.shape[0] / minibatch_size

    # Finally, launch the training loop.
    # We iterate over epochs:
    for epoch_index in range(number_of_epochs):
        # In each epoch_index, we do a full pass over the training data:
        epoch_running_time = 0;

        total_train_loss = 0;
        total_train_accuracy = 0;
        total_train_instances = 0;
        for minibatch_index in xrange(number_of_minibatches):
            minibatch_running_time = timeit.default_timer();

            minibatch_x = train_set_x[minibatch_index * minibatch_size:(minibatch_index + 1) * minibatch_size, :]
            minibatch_y = train_set_y[minibatch_index * minibatch_size:(minibatch_index + 1) * minibatch_size]

            iteration_index = epoch_index * number_of_minibatches + minibatch_index

            learning_rate = initial_learning_rate;
            if learning_rate_decay > 0:
                learning_rate *= (1. / (1. + learning_rate_decay * iteration_index))

            minibatch_average_train_loss, minibatch_average_train_accuracy = train_function(minibatch_x, minibatch_y, learning_rate)

            total_train_loss += minibatch_average_train_loss * minibatch_size;
            total_train_accuracy += minibatch_average_train_accuracy * minibatch_size;
            total_train_instances += minibatch_size;

            epoch_running_time += timeit.default_timer() - minibatch_running_time;

            # And a full pass over the validation data:
            if iteration_index % number_of_minibatches == 0 or (
                        iteration_index % validation_interval == 0 and len(valid_set_y) > 0):
                average_train_accuracy = total_train_accuracy / total_train_instances;
                average_train_loss = total_train_loss / total_train_instances;
                print 'train result: epoch %i, minibatch %i, loss %f, accuracy %f%%' % (epoch_index, minibatch_index, average_train_loss, average_train_accuracy * 100)

                average_validate_loss, average_validate_accuracy = validate_function(valid_set_x, valid_set_y);
                print '\tvalidate result: epoch %i, minibatch %i, loss %f, accuracy %f%%' % (epoch_index, minibatch_index, average_validate_loss, average_validate_accuracy * 100)

                # if we got the best validation score until now
                if average_validate_accuracy > highest_average_validate_accuracy:
                    highest_average_validate_accuracy = average_validate_accuracy
                    best_iteration_index = iteration_index

                    # save the best model
                    print '\tbest model found: epoch %i, minibatch %i, accuracy %f%%' % (epoch_index, minibatch_index, average_validate_accuracy * 100)

                    best_model_file_path = os.path.join(output_directory, 'model.pkl')
                    cPickle.dump(network, open(best_model_file_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL);

                average_test_loss, average_test_accuracy = validate_function(test_set_x, test_set_y);
                print '\t\ttest result: epoch %i, minibatch %i, loss %f, accuracy %f%%' % (epoch_index, minibatch_index, average_test_loss, average_test_accuracy * 100)

        average_train_accuracy = total_train_accuracy / total_train_instances;
        average_train_loss = total_train_loss / total_train_instances;
        print 'train result: epoch %i, duration %fs, loss %f, accuracy %f%%' % (
            epoch_index, epoch_running_time, average_train_loss, average_train_accuracy * 100)

        if snapshot_interval > 0 and (epoch_index + 1) % snapshot_interval == 0:
            model_file_path = os.path.join(output_directory, 'model-%d.pkl' % (epoch_index + 1))
            cPickle.dump(network, open(model_file_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL);

    model_file_path = os.path.join(output_directory, 'model-%d.pkl' % (epoch_index + 1))
    cPickle.dump(network, open(model_file_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL);

    end_train = timeit.default_timer()

    snapshot_index = now.strftime("%y%m%d%H%M%S");
    snapshot_directory = os.path.join(output_directory, snapshot_index);
    assert not os.path.exists(snapshot_directory);
    os.mkdir(snapshot_directory);

    shutil.copy(os.path.join(output_directory, 'model.pkl'), os.path.join(snapshot_directory, 'model.pkl'));
    snapshot_pattern = re.compile(r'^model\-\d+.pkl$');
    for file_name in os.listdir(output_directory):
        if not re.match(snapshot_pattern, file_name):
            continue;
        shutil.move(os.path.join(output_directory, file_name), os.path.join(snapshot_directory, file_name));
    shutil.move(os.path.join(output_directory, 'option.pkl'), os.path.join(snapshot_directory, 'option.pkl'));

    print "Optimization complete..."
    print "Best validation score of %f%% obtained at epoch %i on minibatch %i" % (
        highest_average_validate_accuracy * 100., best_iteration_index / number_of_minibatches,
        best_iteration_index % number_of_minibatches);
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_train - start_train) / 60.))

if __name__ == '__main__':
    launch_resume()
