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

#template_pattern = re.compile(r'(?P<pre_rnn>.*)\[(?P<rnn>.+)\](?P<post_rnn>.*)')

def parse_args():
    parser = optparse.OptionParser()
    parser.set_defaults(# parameter set 1
                        input_directory=None,
                        output_directory=None,
                        # pretrained_model_file=None,
                        
                        # parameter set 2
                        number_of_epochs=-1,
                        minibatch_size=1,
                        snapshot_interval=-1,
                        validation_interval=1000,
                        
                        # parameter set 3
                        learning_rate=1e-2,
                        learning_rate_decay=0,
                        window_size=1,
                        position_offset=-1,
                        sequence_length=100,
                        
                        # parameter set 4
                        # vocabulary_dimension=-1,
                        embedding_dimension=-1,
                        layer_dimensions=None,
                        layer_nonlinearities=None,
                        
                        objective_to_minimize=None,
                        
                        dense_activation_parameters="1",
                        dense_activation_styles="bernoulli",

                        # parameter set 5
                        L1_regularizer_lambdas="0",
                        L2_regularizer_lambdas="0",
                        dae_regularizer_lambdas="0",
                        layer_corruption_levels="0",
                        
                        # parameter set 6
                        number_of_training_data=-1,
                        recurrent_style="elman",
                        recurrent_type="RecurrentLayer"
                        )
    # parameter set 1
    parser.add_option("--input_directory", type="string", dest="input_directory",
                      help="input directory [None]")
    parser.add_option("--output_directory", type="string", dest="output_directory",
                      help="output directory [None]")
    # parser.add_option("--pretrained_model_file", type="string", dest="pretrained_model_file",
                      # help="pretrained model file [None]")

    # parameter set 2
    parser.add_option("--minibatch_size", type="int", dest="minibatch_size",
                      help="mini-batch size [1]")
    parser.add_option("--number_of_epochs", type="int", dest="number_of_epochs",
                      help="number of epochs [-1]")
    parser.add_option("--snapshot_interval", type="int", dest="snapshot_interval",
                      help="snapshot interval in number of epochs [-1]")
    parser.add_option("--validation_interval", type="int", dest="validation_interval",
                      help="validation interval in number of mini-batches [1000]")
    # parser.add_option("--improvement_threshold", type="float", dest="improvement_threshold",
                      # help="improvement threshold [0.01]")
    
    # parameter set 3
    parser.add_option("--learning_rate", type="float", dest="learning_rate",
                      help="learning rate [1e-3]")
    parser.add_option("--learning_rate_decay", type="float", dest="learning_rate_decay",
                      help="learning rate decay [0 - no learning rate decay]")
    parser.add_option("--window_size", type="int", dest="window_size",
                      help="window size [1]");
    parser.add_option("--position_offset", type="int", dest="position_offset",
                      help="position offset of current word in window [-1=window_size/2]");
    parser.add_option("--sequence_length", type="int", dest="sequence_length",
                      help="longest sequnece length for back propagation steps [100]");
    parser.add_option("--objective_to_minimize", type="string", dest="objective_to_minimize",
                      help="objective function to minimize [None], example, 'squared_error' represents the neural network optimizes squared error");
    
    # parameter set 4
    # parser.add_option("--vocabulary_dimension", type="int", dest="vocabulary_dimension",
                      # help="vocabulary size [-1]");
    parser.add_option("--embedding_dimension", type="int", dest="embedding_dimension",
                      help="dimension of word embedding layer [-1]");
    
    parser.add_option("--layer_dimensions", type="string", dest="layer_dimensions",
                      help="dimension of different layer [None], example, '100,[200,500],10' represents 3 layers contains 100, 200, 500, and 10 neurons respectively, where [*] indicates the recurrent layers");
    parser.add_option("--layer_nonlinearities", type="string", dest="layer_nonlinearities",
                      help="activation functions of different layer [None], example, 'tanh,softmax' represents 2 layers with tanh and softmax activation function respectively");
    
    parser.add_option("--dense_activation_parameters", type="string", dest="dense_activation_parameters",
                      help="dropout probability of different layer [1], either one number of a list of numbers, example, '0.2' represents 0.2 dropout rate for all input+hidden layers, or '0.2,0.5' represents 0.2 dropout rate for input layer and 0.5 dropout rate for first hidden layer respectively");
    parser.add_option("--dense_activation_styles", type="string", dest="dense_activation_styles",
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
    parser.add_option("--recurrent_style", type="string", dest="recurrent_style",
                      help="recurrent network style [default=elman, bi-elman]");
    parser.add_option("--recurrent_type", type="string", dest="recurrent_type",
                      help="recurrent layer type [default=RecurrentLayer, LSTMLayer]");
    # parser.add_option("--number_of_pretrain_epochs", type="int", dest="number_of_pretrain_epochs",
                      # help="number of pretrain epochs [0 - no pre-training]");

    parser.add_option("--number_of_training_data", type="int", dest="number_of_training_data",
                      help="training data size [-1]");
                      
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
    #assert(options.snapshot_interval > 0);
    snapshot_interval = options.snapshot_interval;

    # parameter set 3
    assert options.learning_rate > 0;
    initial_learning_rate = options.learning_rate;
    assert options.learning_rate_decay >= 0;
    learning_rate_decay = options.learning_rate_decay;
    assert options.window_size > 0
    #assert options.window_size % 2 == 1;
    window_size = options.window_size;
    position_offset = options.position_offset;
    if position_offset < 0:
        position_offset = window_size / 2;
    assert position_offset>=0 and position_offset<window_size
    assert options.sequence_length > 0;
    sequence_length = options.sequence_length;
    
    # parameter set 4
    # assert options.vocabulary_dimension > 0;
    # vocabulary_dimension = options.vocabulary_dimension
    assert options.embedding_dimension > 0;
    embedding_dimension = options.embedding_dimension

    number_of_dense_layers = 0;
    number_of_recurrent_layers = 0;
    number_of_layers = number_of_dense_layers + number_of_recurrent_layers;

    assert options.layer_dimensions != None
    options_layer_dimensions = options.layer_dimensions + ","
    layer_dimensions = [];
    recurrent_mode = False;
    start_index = 0;
    for end_index in xrange(len(options_layer_dimensions)):
        if options_layer_dimensions[end_index]=="[":
            assert (not recurrent_mode)
            recurrent_mode = True;
            start_index = end_index + 1;
        elif options_layer_dimensions[end_index]=="]":
            assert recurrent_mode
            layer_dimensions.append([int(options_layer_dimensions[start_index:end_index])]);
            recurrent_mode = False;
            start_index = end_index + 1;
        elif options_layer_dimensions[end_index]==",":
            if end_index>start_index:
                if recurrent_mode:
                    layer_dimensions.append([int(options_layer_dimensions[start_index:end_index])]);
                    number_of_recurrent_layers += 1;
                else:
                    layer_dimensions.append(int(options_layer_dimensions[start_index:end_index]));
                    number_of_dense_layers += 1;
            start_index = end_index + 1;

    assert options.layer_nonlinearities != None
    options_layer_nonlinearities = options.layer_nonlinearities + ","
    layer_nonlinearities = [];
    recurrent_mode = False;
    start_index = 0;
    for end_index in xrange(len(options_layer_nonlinearities)):
        if options_layer_nonlinearities[end_index] == "[":
            assert (not recurrent_mode)
            recurrent_mode = True;
            start_index = end_index + 1;
        elif options_layer_nonlinearities[end_index] == "]":
            assert recurrent_mode
            layer_nonlinearities.append([getattr(lasagne.nonlinearities, options_layer_nonlinearities[start_index:end_index])]);
            recurrent_mode = False;
            start_index = end_index + 1;
        elif options_layer_nonlinearities[end_index] == ",":
            if end_index > start_index:
                if recurrent_mode:
                    layer_nonlinearities.append([getattr(lasagne.nonlinearities, options_layer_nonlinearities[start_index:end_index])]);
                else:
                    layer_nonlinearities.append(getattr(lasagne.nonlinearities, options_layer_nonlinearities[start_index:end_index]));
            start_index = end_index + 1;

    '''
    matcher = re.match(template_pattern, options.layer_dimensions);
    assert matcher is not None
    pre_rnn_layer_dimensions = matcher.group("pre_rnn").strip(",");
    rnn_layer_dimensions = matcher.group("rnn").strip(",");
    post_rnn_layer_dimensions = matcher.group("post_rnn").strip(",");
    pre_rnn_layer_dimensions = [] if pre_rnn_layer_dimensions == "" else [int(dimensionality) for dimensionality in pre_rnn_layer_dimensions.split(",")]
    rnn_layer_dimensions = [int(dimensionality) for dimensionality in rnn_layer_dimensions.split(",")]
    post_rnn_layer_dimensions = [] if post_rnn_layer_dimensions == "" else [int(dimensionality) for dimensionality in post_rnn_layer_dimensions.split(",")]

    number_of_layers = len(pre_rnn_layer_dimensions) + len(rnn_layer_dimensions) + len(post_rnn_layer_dimensions);

    assert options.layer_nonlinearities != None
    matcher = re.match(template_pattern, options.layer_nonlinearities);
    assert matcher is not None
    pre_rnn_layer_nonlinearities = matcher.group("pre_rnn").strip(",");
    rnn_layer_nonlinearities = matcher.group("rnn").strip(",");
    post_rnn_layer_nonlinearities = matcher.group("post_rnn").strip(",");
    pre_rnn_layer_nonlinearities = [] if pre_rnn_layer_nonlinearities == "" else [getattr(lasagne.nonlinearities, layer_nonlinearity) for layer_nonlinearity in pre_rnn_layer_nonlinearities.split(",")]
    rnn_layer_nonlinearities = [getattr(lasagne.nonlinearities, layer_nonlinearity) for layer_nonlinearity in rnn_layer_nonlinearities.split(",")]
    post_rnn_layer_nonlinearities = [] if post_rnn_layer_nonlinearities == "" else [getattr(lasagne.nonlinearities, layer_nonlinearity) for layer_nonlinearity in post_rnn_layer_nonlinearities.split(",")]
    
    assert len(pre_rnn_layer_dimensions) == len(pre_rnn_layer_nonlinearities)
    assert len(rnn_layer_dimensions) == len(rnn_layer_nonlinearities)
    assert len(post_rnn_layer_dimensions) == len(post_rnn_layer_nonlinearities)
    '''

    assert options.objective_to_minimize != None
    objective_to_minimize = options.objective_to_minimize;
    objective_to_minimize = getattr(lasagne.objectives, objective_to_minimize)

    dense_activation_styles = options.dense_activation_styles;
    dense_activation_style_tokens = dense_activation_styles.split(",")
    if len(dense_activation_style_tokens) == 1:
        dense_activation_styles = [dense_activation_styles for layer_index in xrange(number_of_dense_layers)]
    elif len(dense_activation_style_tokens) == number_of_dense_layers:
        dense_activation_styles = dense_activation_style_tokens
        # [float(layer_activation_parameter) for layer_activation_parameter in layer_activation_parameter_tokens]
    else:
        sys.stderr.write("error: unrecognized configuration for layer_activation_styles %s\n" % dense_activation_styles);
        sys.exit()
        
    assert len(dense_activation_styles) == number_of_dense_layers;
    for dense_activation_style in dense_activation_styles:
        assert dense_activation_style in set(["bernoulli", "beta_bernoulli", "reciprocal_beta_bernoulli", "reverse_reciprocal_beta_bernoulli", "mixed_beta_bernoulli"])
    
    dense_activation_parameters = options.dense_activation_parameters;
    dense_activation_parameter_tokens = dense_activation_parameters.split(",")
    if len(dense_activation_parameter_tokens) == 1:
        dense_activation_parameters = [dense_activation_parameters for layer_index in xrange(number_of_dense_layers)]
    elif len(dense_activation_parameter_tokens) == number_of_dense_layers:
        dense_activation_parameters = dense_activation_parameter_tokens
        # [float(layer_activation_parameter) for layer_activation_parameter in layer_activation_parameter_tokens]
    else:
        sys.stderr.write("error: unrecognized configuration for layer_activation_parameters %s\n" % dense_activation_parameters);
        sys.exit()
    # assert (layer_activation_parameter >= 0 for layer_activation_parameter in layer_activation_parameters)
    # assert (layer_activation_parameter <= 1 for layer_activation_parameter in layer_activation_parameters)
    
    for layer_index in xrange(number_of_dense_layers):
        if dense_activation_styles[layer_index] == "bernoulli":
            dense_activation_parameters[layer_index] = float(dense_activation_parameters[layer_index])
            assert dense_activation_parameters[layer_index] <= 1;
            assert dense_activation_parameters[layer_index] > 0;
        elif dense_activation_styles[layer_index] == "beta_bernoulli" or dense_activation_styles[layer_index] == "reciprocal_beta_bernoulli" or dense_activation_styles[layer_index] == "reverse_reciprocal_beta_bernoulli" or dense_activation_styles[layer_index] == "mixed_beta_bernoulli":
            dense_activation_parameter_tokens = dense_activation_parameters[layer_index].split("+");
            if len(dense_activation_parameter_tokens) == 1:
                dense_activation_parameters[layer_index] = (float(dense_activation_parameter_tokens[0]), 1.0)
            elif len(dense_activation_parameter_tokens) == 2:
                dense_activation_parameters[layer_index] = (float(dense_activation_parameter_tokens[0]), float(dense_activation_parameter_tokens[1]))
            else:
                sys.stderr.write("error: unrecognized configuration for layer_activation_style %s\n" % dense_activation_styles[layer_index]);
                sys.exit()
            assert dense_activation_parameters[layer_index][0] > 0;
            assert dense_activation_parameters[layer_index][1] > 0;
            
            if dense_activation_styles[layer_index] == "mixed_beta_bernoulli":
                assert dense_activation_parameters[layer_index][0] < 1;
    
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

    '''
    dae_regularizer_lambdas = options.dae_regularizer_lambdas
    dae_regularizer_lambda_tokens = dae_regularizer_lambdas.split(",")
    if len(dae_regularizer_lambda_tokens) == 1:
        dae_regularizer_lambdas = [float(dae_regularizer_lambdas) for layer_index in xrange(number_of_layers - 1)]
    else:
        dae_regularizer_lambdas = [float(dae_regularizer_lambda_token) for dae_regularizer_lambda_token in dae_regularizer_lambda_tokens]
    assert len(dae_regularizer_lambdas) == number_of_layers - 1;
    '''

    # parameter set 6
    '''
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

    # parameter set 6
    recurrent_style = options.recurrent_style;
    assert recurrent_style in ["elman", "bi-elman"]
    recurrent_type = options.recurrent_type
        
    #
    #
    #
    #
    #
    
    data_x = numpy.load(os.path.join(input_directory, "train.feature.npy"))
    data_y = numpy.load(os.path.join(input_directory, "train.label.npy"))
    
    assert len(data_x) == len(data_y);
    for datum_x, datum_y in zip(data_x, data_y):
        assert datum_x.shape == datum_y.shape;

    vocabulary_dimension = 0;
    for line in open(os.path.join(input_directory, "mapping.feature"), 'r'):
        vocabulary_dimension += 1;
    # this is to include a dummy entry for out-of-vocabulary type
    vocabulary_dimension += 1;

    label_dimension = 0;
    for line in open(os.path.join(input_directory, "mapping.label"), 'r'):
        label_dimension += 1;

    # parameter set 6
    # assert(options.number_of_training_data <= 0);
    number_of_training_data = options.number_of_training_data;
    if number_of_training_data <= 0:
        number_of_training_data = len(data_y);
    assert number_of_training_data > 0 and number_of_training_data <= len(data_y)

    indices = range(len(data_y))
    numpy.random.shuffle(indices);
    
    train_set_x = data_x[indices[:number_of_training_data]]
    # train_set_m = data_m[indices[:number_of_training_data]]
    train_set_y = data_y[indices[:number_of_training_data]]

    valid_set_x = data_x[indices[number_of_training_data:]]
    # valid_set_m = data_m[indices[number_of_training_data:]]
    valid_set_y = data_y[indices[number_of_training_data:]]
    
    print "successfully load data with %d for training and %d for validation..." % (train_set_x.shape[0], valid_set_x.shape[0])

    test_set_x = numpy.load(os.path.join(input_directory, "test.feature.npy"))
    test_set_y = numpy.load(os.path.join(input_directory, "test.label.npy"))
    assert len(test_set_x) == len(test_set_y);

    print "successfully load data with %d for testing..." % (len(test_set_x))
    
    #
    #
    #
    #
    #
    
    # create output directory
    now = datetime.datetime.now();
    suffix = now.strftime("%y%m%d-%H%M%S") + "";
    suffix += "-%s" % (recurrent_style);
    suffix += "-%s" % (recurrent_type);
    suffix += "-T%d" % (number_of_training_data);
    #suffix += "-E%d" % (number_of_epochs);
    #suffix += "-S%d" % (snapshot_interval);
    #suffix += "-B%d" % (minibatch_size);
    #suffix += "-aa%f" % (initial_learning_rate);
    # suffix += "-l1r%f" % (L1_regularizer_lambdas);
    # suffix += "-l2r%d" % (L2_regularizer_lambdas);
    suffix += "/";
    
    output_directory = os.path.join(output_directory, suffix);
    os.mkdir(os.path.abspath(output_directory));

    numpy.save(os.path.join(output_directory, "train.index.npy"), indices[:number_of_training_data]);
    numpy.save(os.path.join(output_directory, "valid.index.npy"), indices[number_of_training_data:]);
    
    print "========== ========== ========== ========== =========="
    # parameter set 1
    print "output_directory=" + output_directory
    print "input_directory=" + input_directory
    print "dataset_name=" + dataset_name
    # print "pretrained_model_file=%s" % pretrained_model_file
    # print "dictionary file=" + str(dict_file)
    # parameter set 2
    print "number_of_epochs=%d" % (number_of_epochs);
    print "minibatch_size=" + str(minibatch_size)
    print "snapshot_interval=" + str(snapshot_interval);
    print "validation_interval=%d" % validation_interval;
    
    # parameter set 3
    print "learning_rate=" + str(initial_learning_rate)
    print "window_size=" + str(window_size)
    print "position_offset=" + str(position_offset)
    print "sequence_length=" + str(sequence_length)
    print "objective_to_minimize=%s" % (objective_to_minimize)
    
    # parameter set 4
    print "layer_dimensions=%s" % (layer_dimensions)
    print "layer_nonlinearities=%s" % (layer_nonlinearities)
    #print "layer_dimensions=%s,%s,%s" % (pre_rnn_layer_dimensions, rnn_layer_dimensions, post_rnn_layer_dimensions)
    #print "layer_nonlinearities=%s,%s,%s" % (pre_rnn_layer_nonlinearities, rnn_layer_nonlinearities, post_rnn_layer_nonlinearities)

    print "dense_activation_parameters=%s" % (dense_activation_parameters)
    print "dense_activation_styles=%s" % (dense_activation_styles)
    
    # parameter set 5
    print "L1_regularizer_lambdas=%s" % (L1_regularizer_lambdas)
    print "L2_regularizer_lambdas=%s" % (L2_regularizer_lambdas);
    #print "dae_regularizer_lambdas=%s" % (dae_regularizer_lambdas);
    #print "layer_corruption_levels=%s" % (layer_corruption_levels);
    
    # paramter set 6
    print "number_of_training_data=%d" % (number_of_training_data);
    print "recurrent_style=%s" % (recurrent_style);
    print "recurrent_type=%s" % (recurrent_type);
    print "========== ========== ========== ========== =========="

    cPickle.dump(options, open(os.path.join(output_directory, "option.pkl"), 'wb'), protocol=cPickle.HIGHEST_PROTOCOL);

    ######################
    # BUILD ACTUAL MODEL #
    ######################

    # allocate symbolic variables for the data
    x = theano.tensor.itensor3('x')  # as many columns as context window size/lines as words in the sentence
    # m = theano.tensor.itensor3('m')  # as many columns as context window size/lines as words in the sentence
    # x = theano.tensor.imatrix('x')  # as many columns as context window size/lines as words in the sentence
    m = theano.tensor.imatrix('m')  # as many columns as context window size/lines as words in the sentence
    # y = theano.tensor.imatrix('y')  # label
    y = theano.tensor.ivector('y')  # label
    lr = theano.tensor.scalar('learning_rate');
    
    # input_layer = lasagne.layers.InputLayer(shape=input_shape, input_var=x)
    input_layer = lasagne.layers.InputLayer(shape=(None, sequence_length, window_size,), input_var=x)
    mask_layer = lasagne.layers.InputLayer(shape=(None, sequence_length), input_var=m)
    
    if recurrent_style == "elman":
        import elman
        network = elman.ElmanNetwork(
            input_network=input_layer,
            input_mask=mask_layer,
            vocabulary_dimension=vocabulary_dimension,
            embedding_dimension=embedding_dimension,
            window_size=window_size,
            position_offset=position_offset,
            sequence_length=sequence_length,
            layer_dimensions=layer_dimensions,
            layer_nonlinearities=layer_nonlinearities,
            #
            #
            #
            #
            #
            dense_activation_parameters=dense_activation_parameters,
            dense_activation_styles=dense_activation_styles,
            #
            #
            #
            #
            #
            recurrent_type=recurrent_type,
            objective_to_minimize=objective_to_minimize,
        )
    elif recurrent_style=="bi-elman":
        import bi_elman
        network = bi_elman.BidirectionalElmanNetwork(
            input_network=input_layer,
            input_mask=mask_layer,
            vocabulary_dimension=vocabulary_dimension,
            embedding_dimension=embedding_dimension,
            window_size=window_size,
            position_offset=position_offset,
            sequence_length=sequence_length,
            layer_dimensions=layer_dimensions,
            layer_nonlinearities=layer_nonlinearities,
            #
            #
            #
            #
            #
            dense_activation_parameters=dense_activation_parameters,
            dense_activation_styles=dense_activation_styles,
            #
            #
            #
            #
            #
            objective_to_minimize=objective_to_minimize,
            )
    else:
        sys.stderr.write("Undefined recurrent style %s..." % recurrent_style);
        sys.exit();

    network.set_L1_regularizer_lambda(L1_regularizer_lambdas)
    network.set_L2_regularizer_lambda(L2_regularizer_lambdas)
    #network.set_dae_regularizer_lambda(dae_regularizer_lambdas, layer_corruption_levels)

    ########################
    # BUILD LOSS FUNCTIONS #
    ########################
    
    # Create a train_loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy train_loss):
    train_prediction = network.get_output(dtype=theano.config.floatX)
    train_loss = network.get_objective_to_minimize(y, dtype=theano.config.floatX);
    # train_loss = theano.tensor.mean(lasagne.objectives.categorical_crossentropy(train_prediction, y))
    train_accuracy = theano.tensor.mean(theano.tensor.eq(theano.tensor.argmax(train_prediction, axis=1), y), dtype=theano.config.floatX)
    
    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    all_params = network.get_network_params(trainable=True)
    updates = lasagne.updates.nesterov_momentum(train_loss, all_params, lr, momentum=0.95)
    
    # Create a train_loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the networks,
    # disabling dropout layers.
    validate_prediction = network.get_output(deterministic=True, dtype=theano.config.floatX)
    validate_loss = network.get_objective_to_minimize(y, deterministic=True, dtype=theano.config.floatX);
    #validate_loss = theano.tensor.mean(theano.tensor.nnet.categorical_crossentropy(validate_prediction, y), dtype=theano.config.floatX)
    # As a bonus, also create an expression for the classification accuracy:
    validate_accuracy = theano.tensor.mean(theano.tensor.eq(theano.tensor.argmax(validate_prediction, axis=1), y), dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training train_loss:
    train_function = theano.function(
        inputs=[x, y, m, lr],
        outputs=[train_loss, train_accuracy],
        updates=updates
    )
    
    # Compile a second function computing the validation train_loss and accuracy:
    validate_function = theano.function(
        inputs=[x, y, m],
        outputs=[validate_loss, validate_accuracy],
    )

    ########################
    # START MODEL TRAINING #
    ########################
    
    highest_average_validate_accuracy = 0
    best_iteration_index = 0
    
    start_train = timeit.default_timer()
    
    #model_file_path = os.path.join(output_directory, 'model-%d.pkl' % (0))
    #cPickle.dump(network, open(model_file_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL);
    
    # compute number of minibatches for training, validation and testing
    #number_of_minibatches = train_set_x.get_value(borrow=True).shape[0] / minibatch_size
    number_of_minibatches = train_set_x.shape[0] / minibatch_size

    #
    #
    #
    #
    #

    '''
    # Parse train data into sequences
    train_sequence_x = -numpy.ones((0, sequence_length, window_size), dtype=numpy.int32);
    train_sequence_m = numpy.zeros((0, sequence_length), dtype=numpy.int8);
    train_sequence_y = numpy.zeros(0, dtype=numpy.int32);

    train_sequence_indices_by_instance = [0];
    for train_instance_x, train_instance_y in zip(train_set_x, train_set_y):
        # context_windows = get_context_windows(train_sequence_x, window_size)
        # train_minibatch, train_minibatch_masks = get_mini_batches(context_windows, backprop_step);
        instance_sequence_x, instance_sequence_m = network.get_instance_sequences(train_instance_x);
        assert len(instance_sequence_x) == len(instance_sequence_m);
        assert len(instance_sequence_x) == len(train_instance_y);
        # print mini_batches.shape, mini_batch_masks.shape, train_sequence_y.shape

        train_sequence_x = numpy.concatenate((train_sequence_x, instance_sequence_x), axis=0);
        train_sequence_m = numpy.concatenate((train_sequence_m, instance_sequence_m), axis=0);
        train_sequence_y = numpy.concatenate((train_sequence_y, train_instance_y), axis=0);

        train_sequence_indices_by_instance.append(len(train_sequence_y));

    # Parse validate data into sequences
    valid_sequence_x = -numpy.ones((0, sequence_length, window_size), dtype=numpy.int32);
    valid_sequence_m = numpy.zeros((0, sequence_length), dtype=numpy.int8);
    valid_sequence_y = numpy.zeros(0, dtype=numpy.int32);

    valid_sequence_indices_by_instance = [0];
    for valid_instance_x, valid_instance_y in zip(valid_set_x, valid_set_y):
        instance_sequence_x, instance_sequence_m = network.get_instance_sequences(valid_instance_x);
        assert len(instance_sequence_x) == len(instance_sequence_m);
        assert len(instance_sequence_x) == len(valid_instance_y);

        valid_sequence_x = numpy.concatenate((valid_sequence_x, instance_sequence_x), axis=0);
        valid_sequence_m = numpy.concatenate((valid_sequence_m, instance_sequence_m), axis=0);
        valid_sequence_y = numpy.concatenate((valid_sequence_y, valid_instance_y), axis=0);

        valid_sequence_indices_by_instance.append(len(valid_sequence_y));

    # Parse test data into sequences
    test_sequence_x = -numpy.ones((0, sequence_length, window_size), dtype=numpy.int32);
    test_sequence_m = numpy.zeros((0, sequence_length), dtype=numpy.int8);
    test_sequence_y = numpy.zeros(0, dtype=numpy.int32);

    test_sequence_indices_by_instance = [0];
    for test_instance_x, test_instance_y in zip(test_set_x, test_set_y):
        instance_sequence_x, instance_sequence_m = network.get_instance_sequences(test_instance_x);
        assert len(instance_sequence_x) == len(instance_sequence_m);
        assert len(instance_sequence_x) == len(test_instance_y);

        test_sequence_x = numpy.concatenate((test_sequence_x, instance_sequence_x), axis=0);
        test_sequence_m = numpy.concatenate((test_sequence_m, instance_sequence_m), axis=0);
        test_sequence_y = numpy.concatenate((test_sequence_y, test_instance_y), axis=0);

        test_sequence_indices_by_instance.append(len(test_sequence_y));
    '''

    train_sequence_x, train_sequence_m, train_sequence_y, train_sequence_indices_by_instance = network.parse_sequence(train_set_x, train_set_y)
    valid_sequence_x, valid_sequence_m, valid_sequence_y, valid_sequence_indices_by_instance = network.parse_sequence(valid_set_x, valid_set_y)
    test_sequence_x, test_sequence_m, test_sequence_y, test_sequence_indices_by_instance = network.parse_sequence(test_set_x, test_set_y)

    #
    #
    #
    #
    #
    
    # Finally, launch the training loop.
    # We iterate over epochs:
    for epoch_index in range(number_of_epochs):
        # In each epoch_index, we do a full pass over the training data:
        epoch_running_time = 0;

        total_train_loss = 0;
        total_train_accuracy = 0;
        for minibatch_index in xrange(number_of_minibatches):
            minibatch_running_time = timeit.default_timer();

            iteration_index = epoch_index * number_of_minibatches + minibatch_index

            instance_start_index = minibatch_index * minibatch_size;
            instance_end_index = (minibatch_index + 1) * minibatch_size;

            train_sequence_start_index = train_sequence_indices_by_instance[instance_start_index];
            train_sequence_end_index = train_sequence_indices_by_instance[instance_end_index];

            learning_rate = initial_learning_rate;
            if learning_rate_decay>0:
                learning_rate *= (1. / (1. + learning_rate_decay * iteration_index))

            minibatch_average_train_loss, minibatch_average_train_accuracy = train_function(
                train_sequence_x[train_sequence_start_index:train_sequence_end_index, :, :],
                train_sequence_y[train_sequence_start_index:train_sequence_end_index],
                train_sequence_m[train_sequence_start_index:train_sequence_end_index, :],
                learning_rate);
            
            #embedding_layer = [layer for layer in network.get_all_layers() if isinstance(layer, lasagne.layers.EmbeddingLayer)][0];
            #print numpy.sum(embedding_layer.W.eval()**2)
            #print numpy.sum(network._embeddings.eval()**2)
            #normalize_embedding_function();
            network._normalize_embeddings_function();
            #print numpy.sum(network._embeddings.eval()**2)
            #new_values = network._embeddings.eval();
            
            total_train_loss += minibatch_average_train_loss * (train_sequence_end_index - train_sequence_start_index);
            total_train_accuracy += minibatch_average_train_accuracy * (train_sequence_end_index - train_sequence_start_index);

            epoch_running_time += timeit.default_timer() - minibatch_running_time;

            if iteration_index % 1000 == 0: # or train_sequence_end_index % 1000 == 0:
                print "train progress: %d sequences by %d minibatches" % (train_sequence_end_index, iteration_index+1)

            #
            #
            #
            #
            #

            # And a full pass over the validation data:
            if iteration_index % number_of_minibatches == 0 or (iteration_index % validation_interval == 0 and len(valid_set_y) > 0):

                #
                #
                #
                #
                #

                total_validate_loss = 0;
                total_validate_accuracy = 0;
                for valid_instance_index in xrange(len(valid_set_y)):
                    valid_sequence_start_index = valid_sequence_indices_by_instance[valid_instance_index];
                    valid_sequence_end_index = valid_sequence_indices_by_instance[valid_instance_index + 1];

                    minibatch_validate_loss, minibatch_validate_accuracy = validate_function(
                        valid_sequence_x[valid_sequence_start_index:valid_sequence_end_index, :, :],
                        valid_sequence_y[valid_sequence_start_index:valid_sequence_end_index],
                        valid_sequence_m[valid_sequence_start_index:valid_sequence_end_index, :])

                    total_validate_loss += minibatch_validate_loss * (valid_sequence_end_index - valid_sequence_start_index);
                    total_validate_accuracy += minibatch_validate_accuracy * (valid_sequence_end_index - valid_sequence_start_index);

                    if valid_instance_index % 1000 == 0: # or valid_sequence_end_index % 1000 == 0:
                        print "\tvalidate progress: %d sequences by %d instances" % (valid_sequence_end_index+1, valid_instance_index+1)

                average_validate_loss = total_validate_loss / valid_sequence_end_index;
                average_validate_accuracy = total_validate_accuracy / valid_sequence_end_index;
                print '\tvalidate result: epoch %i, minibatch %i, loss %f, accuracy %f%%' % (epoch_index, minibatch_index, average_validate_loss, average_validate_accuracy * 100)
                        
                # if we got the best validation score until now
                if average_validate_accuracy > highest_average_validate_accuracy:
                    highest_average_validate_accuracy = average_validate_accuracy
                    best_iteration_index = iteration_index
                    
                    # save the best model
                    print '\tbest model found: epoch %i, minibatch %i, accuracy %f%%' % (epoch_index+1, minibatch_index+1, average_validate_accuracy * 100)

                    best_model_file_path = os.path.join(output_directory, 'model.pkl')
                    cPickle.dump(network, open(best_model_file_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL);

                #
                #
                #
                #
                #

                total_test_loss = 0;
                total_test_accuracy = 0;
                for test_instance_index in xrange(len(test_set_y)):
                    test_sequence_start_index = test_sequence_indices_by_instance[test_instance_index];
                    test_sequence_end_index = test_sequence_indices_by_instance[test_instance_index + 1];

                    minibatch_test_loss, minibatch_test_accuracy = validate_function(
                        test_sequence_x[test_sequence_start_index:test_sequence_end_index, :, :],
                        test_sequence_y[test_sequence_start_index:test_sequence_end_index],
                        test_sequence_m[test_sequence_start_index:test_sequence_end_index, :])

                    total_test_loss += minibatch_test_loss * (test_sequence_end_index - test_sequence_start_index);
                    total_test_accuracy += minibatch_test_accuracy * (test_sequence_end_index - test_sequence_start_index);

                    if test_instance_index % 1000 == 0: # or test_sequence_end_index % 1000 == 0:
                        print "\t\ttest progress: %d sequences by %d instances" % (test_sequence_end_index+1, test_instance_index+1)
                        
                average_test_loss = total_test_loss / test_sequence_end_index;
                average_test_accuracy = total_test_accuracy / test_sequence_end_index;
                print '\t\ttest result: epoch %i, minibatch %i, loss %f, accuracy %f%%' % (epoch_index, minibatch_index, average_test_loss, average_test_accuracy * 100)

        average_train_loss = total_train_loss / train_sequence_end_index
        average_train_accuracy = total_train_accuracy / train_sequence_end_index
        print 'train result: epoch %i, duration %fs, loss %f, accuracy %f%%' % (
            epoch_index, epoch_running_time, average_train_loss, average_train_accuracy * 100)

        if snapshot_interval>0 and (epoch_index + 1) % snapshot_interval == 0:
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

'''
def parse_sequence(set_x, set_y, sequence_length, window_size):
    # Parse train data into sequences
    sequence_x = -numpy.ones((0, sequence_length, window_size), dtype=numpy.int32);
    sequence_m = numpy.zeros((0, sequence_length), dtype=numpy.int8);
    sequence_y = numpy.zeros(0, dtype=numpy.int32);

    sequence_indices_by_instance = [0];
    for instance_x, instance_y in zip(set_x, set_y):
        # context_windows = get_context_windows(train_sequence_x, window_size)
        # train_minibatch, train_minibatch_masks = get_mini_batches(context_windows, backprop_step);
        instance_sequence_x, instance_sequence_m = network.get_instance_sequences(instance_x);
        assert len(instance_sequence_x) == len(instance_sequence_m);
        assert len(instance_sequence_x) == len(instance_y);
        # print mini_batches.shape, mini_batch_masks.shape, train_sequence_y.shape

        sequence_x = numpy.concatenate((sequence_x, instance_sequence_x), axis=0);
        sequence_m = numpy.concatenate((sequence_m, instance_sequence_m), axis=0);
        sequence_y = numpy.concatenate((sequence_y, instance_y), axis=0);

        sequence_indices_by_instance.append(len(sequence_y));
'''

"""
def get_context_windows(sequence, window_size, vocab_size=None):
    '''
    window_size :: int corresponding to the size of the window
    given a list of indexes composing a sentence
    it will return a list of list of indexes corresponding
    to context windows surrounding each word in the sentence
    '''
    assert (window_size % 2) == 1
    assert window_size >= 1
    sequence = list(sequence)

    if vocab_size == None:
        context_windows = -numpy.ones((len(sequence), window_size), dtype=numpy.int32);
        padded_sequence = window_size / 2 * [-1] + sequence + window_size / 2 * [-1]
        for i in xrange(len(sequence)):
            context_windows[i, :] = padded_sequence[i:i + window_size];
    else:
        context_windows = numpy.zeros((len(sequence), vocab_size), dtype=numpy.int32);
        padded_sequence = window_size / 2 * [-1] + sequence + window_size / 2 * [-1]
        for i in xrange(len(sequence)):
            for j in padded_sequence[i:i + window_size]:
                context_windows[i, j] += 1;

    # assert len(context_windows) == len(sequence)
    return context_windows

def get_mini_batches(context_windows, backprop_step):
    '''
    context_windows :: list of word idxs
    return a list of minibatches of indexes
    which size is equal to backprop_step
    border cases are treated as follow:
    eg: [0,1,2,3] and backprop_step = 3
    will output:
    [[0],[0,1],[0,1,2],[1,2,3]]
    '''

    '''
    mini_batches = [context_windows[:i] for i in xrange(1, min(backprop_step, len(context_windows) + 1))]
    mini_batches += [context_windows[i - backprop_step:i] for i in xrange(backprop_step, len(context_windows) + 1) ]
    assert len(context_windows) == len(mini_batches)
    return mini_batches
    '''

    sequence_length, window_size = context_windows.shape;
    mini_batches = -numpy.ones((sequence_length, backprop_step, window_size), dtype=numpy.int32);
    mini_batch_masks = numpy.zeros((sequence_length, backprop_step), dtype=numpy.int32);
    for i in xrange(min(sequence_length, backprop_step)):
        mini_batches[i, 0:i + 1, :] = context_windows[0:i + 1, :];
        mini_batch_masks[i, 0:i + 1] = 1;
    for i in xrange(min(sequence_length, backprop_step), sequence_length):
        mini_batches[i, :, :] = context_windows[i - backprop_step + 1:i + 1, :];
        mini_batch_masks[i, :] = 1;
    return mini_batches, mini_batch_masks
"""

if __name__ == '__main__':
    launch_train()
