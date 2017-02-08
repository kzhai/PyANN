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

def parse_args():
    parser = optparse.OptionParser()
    parser.set_defaults(# parameter set 1
                        input_directory=None,
                        output_directory=None,

                        # parameter set 2
                        number_of_epochs=-1,
                        minibatch_size=1,
                        snapshot_interval=10,
                        # validation_interval=1000,
                        
                        # parameter set 3
                        learning_rate=1e-3,
                        number_of_gibbs_steps=1,
                        persistent=False,
                        # objective_to_minimize="None",

                        # parameter set 4
                        layer_dimensions=None,
                        # layer_nonlinearities=None,
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
    # parser.add_option("--validation_interval", type="int", dest="validation_interval",
                      # help="validation interval in number of mini-batches [1000]");                      
    
    # parameter set 3
    parser.add_option("--learning_rate", type="float", dest="learning_rate",
                      help="learning rate [1e-3]")
    parser.add_option("--number_of_gibbs_steps", type="int", dest="number_of_gibbs_steps",
                      help="number_of_gibbs_steps [1]")
    parser.add_option("--persistent", action="store_true", dest="persistent", default=False,
                      help="persistent contrastive divergence [False]")
    # parser.add_option("--objective_to_minimize", type="string", dest="objective_to_minimize",
                      # help="objective function to minimize [None], example, 'squared_error' represents the neural network optimizes squared error");
    
    # parameter set 4
    parser.add_option("--layer_dimensions", type="string", dest="layer_dimensions",
                      help="dimension of different layer [None], example, '100,500,10' represents 3 layers contains 100, 500, and 10 neurons respectively");
    # parser.add_option("--layer_nonlinearities", type="string", dest="layer_nonlinearities",
                      # help="activation functions of different layer [None], example, 'tanh,softmax' represents 2 layers with tanh and softmax activation function respectively");
                    
    # parser.add_option("--layer_activation_parameters", type="string", dest="layer_activation_parameters",
                      # help="dropout probability of different layer [1], either one number of a list of numbers, example, '0.2' represents 0.2 dropout rate for all input+hidden layers, or '0.2,0.5' represents 0.2 dropout rate for input layer and 0.5 dropout rate for first hidden layer respectively");
    # parser.add_option("--layer_latent_feature_alphas", type="string", dest="layer_latent_feature_alphas",
                      # help="alpha for latent feature ");
    # parser.add_option("--layer_activation_styles", type="string", dest="layer_activation_styles",
                      # help="dropout style different layer [bernoulli], example, 'bernoulli,beta-bernoulli' represents 2 layers with bernoulli and beta-bernoulli dropout respectively");
                      
    # parameter set 5
    # parser.add_option("--L1_regularizer_lambdas", type="string", dest="L1_regularizer_lambdas",
                      # help="L1 regularization lambda [0]")
    # parser.add_option("--L2_regularizer_lambdas", type="string", dest="L2_regularizer_lambdas",
                      # help="L2 regularization lambda [0]")
    # parser.add_option("--layer_corruption_levels", type="string", dest="layer_corruption_levels",
                      # help="layer corruption level for pre-training [0], either one number of a list of numbers, example, '0.2' represents 0.2 corruption level for all denoising auto encoders, or '0.2,0.5' represents 0.2 corruption level for first denoising auto encoder layer and 0.5 for second one respectively");
    # parser.add_option("--dae_regularizer_lambdas", type="string", dest="dae_regularizer_lambdas",
                      # help="dae regularization lambda [0]")
    # parser.add_option("--number_of_pretrain_epochs", type="int", dest="number_of_pretrain_epochs",
                      # help="number of pretrain epochs [0 - no pre-training]");
                      
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
    assert(options.snapshot_interval > 0);
    snapshot_interval = options.snapshot_interval;
    # assert(options.validation_interval > 0);
    # validation_interval = options.validation_interval;
    
    # parameter set 3
    assert options.learning_rate > 0;
    learning_rate = options.learning_rate;
    assert options.number_of_gibbs_steps > 0;
    number_of_gibbs_steps = options.number_of_gibbs_steps;
    persistent = options.persistent;
    # assert options.objective_to_minimize != None
    # objective_to_minimize = options.objective_to_minimize;
    # objective_to_minimize = getattr(lasagne.objectives, objective_to_minimize)    
    
    # parameter set 4
    assert options.layer_dimensions != None
    layer_dimensions = [int(dimensionality) for dimensionality in options.layer_dimensions.split(",")]
    number_of_layers = len(layer_dimensions);

    # assert options.layer_nonlinearities != None
    # layer_nonlinearities = options.layer_nonlinearities.split(",")
    # layer_nonlinearities = [getattr(lasagne.nonlinearities, layer_nonlinearity) for layer_nonlinearity in layer_nonlinearities]
    # assert len(layer_nonlinearities) == number_of_layers;
    
    '''
    layer_activation_styles = options.layer_activation_styles;
    layer_activation_styles = layer_activation_styles.split(",")
    if len(layer_activation_styles) == 1:
        layer_activation_styles = [layer_activation_styles for layer_index in xrange(number_of_layers)]
    assert len(layer_activation_styles) == number_of_layers;
    for layer_activation_style in layer_activation_styles:
        assert layer_activation_style in set(["bernoulli", "beta_bernoulli", "reciprocal_beta_bernoulli"])
    
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
        elif layer_activation_styles[layer_index] == "beta_bernoulli" or layer_activation_styles[layer_index] == "reciprocal_beta_bernoulli":
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

    layer_corruption_levels = options.layer_corruption_levels;
    layer_corruption_level_tokens = layer_corruption_levels.split(",")
    if len(layer_corruption_level_tokens) == 1:
        layer_corruption_levels = [float(layer_corruption_levels) for layer_index in xrange(number_of_layers)]
    elif len(layer_corruption_level_tokens) == number_of_layers:
        layer_corruption_levels = [float(layer_corruption_level) for layer_corruption_level in layer_corruption_level_tokens]
    else:
        sys.stderr.write("error: unrecognized configuration for layer_corruption_levels %s\n" % layer_corruption_levels);
        sys.exit()
    assert len(layer_corruption_levels) == number_of_layers;
    assert (layer_corruption_level >= 0 for layer_corruption_level in layer_corruption_levels)
    assert (layer_corruption_level < 1 for layer_corruption_level in layer_corruption_levels)
    '''
    
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
    # data_x = numpy.asarray(data_x, numpy.float32) / 256
    # data_x = data_x / numpy.float32(256)
    # data_x = (data_x - numpy.float32(128)) / numpy.float32(128)
    assert data_x.shape[0] == len(data_y);
    
    input_shape = list(data_x.shape[1:])
    input_shape.insert(0, None)
    
    # number_of_train = int(round(0.85 * len(data_y)));
    indices = range(len(data_y))
    numpy.random.shuffle(indices);
    
    # train_set_x = data_x[indices[:number_of_train], :]
    # train_set_y = data_y[indices[:number_of_train]]

    # valid_set_x = data_x[indices[number_of_train:], :]
    # valid_set_y = data_y[indices[number_of_train:]]
    
    # print "successfully load data with %d for training and %d for validation..." % (train_set_x.shape[0], valid_set_x.shape[0])
    
    train_set_x = data_x[indices]
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
    suffix += "-%s" % ("dbn");
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
    
    # parameter set 2
    options_output_file.write("number_of_epochs=%d\n" % (number_of_epochs));
    options_output_file.write("minibatch_size=" + str(minibatch_size) + "\n");
    options_output_file.write("snapshot_interval=" + str(snapshot_interval) + "\n");
    
    # parameter set 3
    options_output_file.write("learning_rate=" + str(learning_rate) + "\n");
    options_output_file.write("number_of_gibbs_steps=" + str(number_of_gibbs_steps) + "\n");
    options_output_file.write("persistent=" + str(persistent) + "\n");
    # options_output_file.write("objective_to_minimize=%s\n" % (objective_to_minimize));
    
    # parameter set 4
    options_output_file.write("layer_dimensions=%s\n" % (layer_dimensions));
    # options_output_file.write("layer_nonlinearities=%s\n" % (layer_nonlinearities));
    
    # options_output_file.write("layer_activation_parameters=%s\n" % (layer_activation_parameters));
    # options_output_file.write("layer_activation_styles=%s\n" % (layer_activation_styles));
    
    # parameter set 5
    # options_output_file.write("L1_regularizer_lambdas=%s\n" % (L1_regularizer_lambdas));
    # options_output_file.write("L2_regularizer_lambdas=%s\n" % (L2_regularizer_lambdas));
    # options_output_file.write("layer_corruption_levels=%s\n" % (layer_corruption_levels));
    # options_output_file.write("dae_regularizer_lambdas=%s\n" % (dae_regularizer_lambdas));
    # options_output_file.write("number_of_pretrain_epochs=%s\n" % (number_of_pretrain_epochs));
    
    options_output_file.close()
    
    print "========== ========== ========== ========== =========="
    # parameter set 1
    print "output_directory=" + output_directory
    print "input_directory=" + input_directory
    print "dataset_name=" + dataset_name

    # parameter set 2
    print "number_of_epochs=%d" % (number_of_epochs);
    print "snapshot_interval=" + str(snapshot_interval);
    print "minibatch_size=" + str(minibatch_size)
    
    # parameter set 3
    print "learning_rate=" + str(learning_rate)
    print "number_of_gibbs_steps=" + str(number_of_gibbs_steps);
    print "persistent=" + str(persistent);
    # print "objective_to_minimize=%s" % (objective_to_minimize)
    
    # parameter set 4
    print "layer_dimensions=%s" % (layer_dimensions)
    # print "layer_nonlinearities=%s" % (layer_nonlinearities)
    
    # print "layer_activation_parameters=%s" % (layer_activation_parameters)
    # print "layer_activation_styles=%s" % (layer_activation_styles)
    
    # parameter set 5
    # print "L1_regularizer_lambdas=%s" % (L1_regularizer_lambdas)
    # print "L2_regularizer_lambdas=%s" % (L2_regularizer_lambdas);
    # print "layer_corruption_levels=%s" % (layer_corruption_levels);
    # print "dae_regularizer_lambdas=%s" % (dae_regularizer_lambdas);
    # print "number_of_pretrain_epochs=%s" % (number_of_pretrain_epochs);
    print "========== ========== ========== ========== =========="
    
    ######################
    # BUILD ACTUAL MODEL #
    ######################

    # allocate symbolic variables for the data
    x = theano.tensor.matrix('x')  # the data is presented as rasterized images
    # y = theano.tensor.ivector('y')  # the labels are presented as 1D vector of [int] labels
    
    input_layer = lasagne.layers.InputLayer(shape=input_shape, input_var=x)
    
    import dbn
    network = dbn.DeepBeliefNetwork(
        input_network=input_layer,
        layer_dimensions=layer_dimensions,
        # layer_nonlinearities=layer_nonlinearities,
        # layer_corruption_levels=layer_corruption_levels,
        # L1_regularizer_lambdas=L1_regularizer_lambdas,
        # L2_regularizer_lambdas=L2_regularizer_lambdas,
        # objective_to_minimize=objective_to_minimize,
        )
    
    restricted_boltzmann_machines = network.get_restricted_boltzmann_machines();
    train_functions = [];
    for rbm_index, restricted_boltzmann_machine in enumerate(restricted_boltzmann_machines):
        if persistent:
            persistent_chain = theano.shared(numpy.zeros((minibatch_size, layer_dimensions[rbm_index]), dtype=theano.config.floatX), borrow=True)
        else:
            persistent_chain = None;
            
        # get the cost and the gradient corresponding to one step of CD-15
        cost, sample, updates = restricted_boltzmann_machine.get_cost_updates(learning_rate=learning_rate, persistent=persistent_chain, k=number_of_gibbs_steps)
        
        '''
        # Create a train_loss expression for training, i.e., a scalar objective we want
        # to minimize (for our multi-class problem, it is the cross-entropy train_loss):
        train_prediction = network.get_output()
        train_loss = network.get_objective_to_minimize();
        # train_accuracy = theano.tensor.mean(theano.tensor.eq(theano.tensor.argmax(train_prediction, axis=1), y), dtype=theano.config.floatX)
        
        # Create update expressions for training, i.e., how to modify the
        # parameters at each training step. Here, we'll use Stochastic Gradient
        # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
        all_params = network.get_all_params(trainable=True)
        updates = lasagne.updates.nesterov_momentum(train_loss, all_params, learning_rate, momentum=0.95)
    
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
            outputs=[cost, sample],
            updates=updates
            # givens={x: train_set_x[index * batch_size: (index + 1) * batch_size]},
            # name='train_rbm'
        )
        
        train_functions.append(train_function);

    ###############
    # TRAIN MODEL #
    ###############

    start_time = timeit.default_timer()
    
    model_file_path = os.path.join(output_directory, 'model-%d.pkl' % (0))
    cPickle.dump(network, open(model_file_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL);
    
    number_of_minibatches = train_set_x.shape[0] / minibatch_size
    
    for rbm_index in xrange(len(restricted_boltzmann_machines)):
        for epoch_index in xrange(number_of_epochs):
            clock_epoch = time.time()
            
            average_train_losses = []
            for minibatch_index in xrange(number_of_minibatches):
                iteration_index = epoch_index * number_of_minibatches + minibatch_index
                minibatch_x = train_set_x[minibatch_index * minibatch_size:(minibatch_index + 1) * minibatch_size, :]
                
                function_output = train_functions[rbm_index](minibatch_x)
                average_train_loss = function_output[0];
                
                average_train_losses.append(average_train_loss)
            
            clock_epoch = time.time() - clock_epoch;
            
            print 'layer %i, epoch %d, average train loss %f, time elapsed %f' % (rbm_index + 1, epoch_index, numpy.mean(average_train_losses), clock_epoch)

        model_file_path = os.path.join(output_directory, 'model-layer-%d.pkl' % (rbm_index + 1))
        cPickle.dump(network, open(model_file_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL);
        
    model_file_path = os.path.join(output_directory, 'model.pkl')
    cPickle.dump(network, open(model_file_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL);

    end_time = timeit.default_timer()
    print "Optimization complete..."
    print >> sys.stderr, ('The code for file ' + 
                          os.path.split(__file__)[1] + 
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    
if __name__ == '__main__':
    launch_train()
