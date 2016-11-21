import theano
import theano.tensor as T
import numpy as np

from lasagne.layers import Layer
from lasagne import init
from lasagne import nonlinearities

from theano.tensor.shared_randomstreams import RandomStreams
import lasagne

import network

class BidirectionalRecurrentLayer(Layer):
    """Bidirectional Recurrent Layer (BiRNN)
    """
    
    def __init__(self,
                 incoming,
                 num_units,
                 forward_nonlinearity=nonlinearities.sigmoid,
                 backward_nonlinearity=nonlinearities.sigmoid,
                 mask_input=None,
                 W_forward_in_to_hid=lasagne.init.GlorotUniform(gain=network.GlorotUniformGain[nonlinearities.sigmoid]),
                 W_forward_hid_to_hid=lasagne.init.GlorotUniform(gain=network.GlorotUniformGain[nonlinearities.sigmoid]),
                 b_forward_in=lasagne.init.Constant(0.),
                 b_forward_hid=lasagne.init.Constant(0.),
                 W_backward_in_to_hid=lasagne.init.GlorotUniform(gain=network.GlorotUniformGain[nonlinearities.sigmoid]),
                 W_backward_hid_to_hid=lasagne.init.GlorotUniform(gain=network.GlorotUniformGain[nonlinearities.sigmoid]),
                 b_backward_in=lasagne.init.Constant(0.),
                 b_backward_hid=lasagne.init.Constant(0.),
                 **kwargs):
        '''
        W_encoder=init.GlorotUniform(gain=4.0),
        W_decoder=None,
        b_encoder=init.Constant(0.),
        b_decoder=init.Constant(0.),
        '''

        super(BidirectionalRecurrentLayer, self).__init__(incoming, **kwargs)

        self.num_units = num_units

        num_inputs = int(np.prod(self.input_shape[2:]))

        forward_rnn_layer = lasagne.layers.RecurrentLayer(self.input_layer,
                                                          num_units,
                                                          W_in_to_hid=W_forward_in_to_hid,
                                                          W_hid_to_hid=W_forward_hid_to_hid,
                                                          b=b_forward_in,
                                                          nonlinearity=forward_nonlinearity,
                                                          hid_init=b_forward_hid,
                                                          backwards=False,
                                                          learn_init=False,
                                                          gradient_steps=-1,
                                                          grad_clipping=0,
                                                          unroll_scan=False,
                                                          precompute_input=True,
                                                          mask_input=mask_input,
                                                          # only_return_final=True
                                                          );

        backward_rnn_layer = lasagne.layers.RecurrentLayer(self.input_layer,
                                                           num_units,
                                                           W_in_to_hid=W_backward_in_to_hid,
                                                           W_hid_to_hid=W_backward_hid_to_hid,
                                                           b=b_backward_in,
                                                           nonlinearity=backward_nonlinearity,
                                                           hid_init=b_backward_hid,
                                                           backwards=True,
                                                           learn_init=False,
                                                           gradient_steps=-1,
                                                           grad_clipping=0,
                                                           unroll_scan=False,
                                                           precompute_input=True,
                                                           mask_input=mask_input,
                                                           # only_return_final=True
                                                           );

        lasagne.layers.ConcatLayer([forward_rnn_layer, backward_rnn_layer]);

        self.W_forward_in_to_hid = self.add_param(W_forward_in_to_hid, (num_inputs, num_units),
                                                  name="W_forward_in_to_hid");
        self.W_forward_hid_to_hid = self.add_param(W_forward_hid_to_hid, (num_units, num_units),
                                                   name="W_forward_hid_to_hid");
        self.b_forward_in = self.add_param(b_forward_in, (num_units,), name="b_forward_in", regularizable=False);
        self.b_forward_hid = self.add_param(b_forward_hid, (num_units,), name="b_forward_hid", regularizable=False);

        self.W_backward_in_to_hid = self.add_param(W_backward_in_to_hid, (num_inputs, num_units),
                                                   name="W_backward_in_to_hid");
        self.W_backward_hid_to_hid = self.add_param(W_backward_hid_to_hid, (num_units, num_units),
                                                    name="W_backward_hid_to_hid");
        self.b_backward_in = self.add_param(b_backward_in, (num_units,), name="b_backward_in", regularizable=False);
        self.b_backward_hid = self.add_param(b_backward_hid, (num_units,), name="b_backward_hid", regularizable=False);

    def get_output_shape_for(self, input_shape):
        return input_shape[:2] + (self.num_units,)

    def get_output_for(self, input, **kwargs):
        activation = T.dot(input, self.W)
        if self.b is not None:
            activation = activation + self.b
        return self.nonlinearity(activation)

if __name__ == '__main__':
    import numpy
    import os
    import sys

    input_directory=sys.argv[1]
    backprop_step = 5
    window_size = 3
    embedding_dimension = 10;

    data_x = numpy.load(os.path.join(input_directory, "train.feature.npy"))
    data_y = numpy.load(os.path.join(input_directory, "train.label.npy"))

    train_set_x = data_x
    # train_set_m = data_m[indices[:number_of_training_data]]
    train_set_y = data_y

    vocabulary_dimension = 0;
    for line in open(os.path.join(input_directory, "mapping.feature"), 'r'):
        vocabulary_dimension += 1;
    # this is to include a dummy entry for out-of-vocabulary type
    vocabulary_dimension += 1;












    # allocate symbolic variables for the data
    x = theano.tensor.itensor3('x')  # as many columns as context window size/lines as words in the sentence
    # m = theano.tensor.itensor3('m')  # as many columns as context window size/lines as words in the sentence
    # x = theano.tensor.imatrix('x')  # as many columns as context window size/lines as words in the sentence
    m = theano.tensor.imatrix('m')  # as many columns as context window size/lines as words in the sentence
    # y = theano.tensor.imatrix('y')  # label
    y = theano.tensor.ivector('y')  # label

    # input_layer = lasagne.layers.InputLayer(shape=input_shape, input_var=x)
    input_layer = lasagne.layers.InputLayer(shape=(None, backprop_step, window_size,), input_var=x)
    mask_layer = lasagne.layers.InputLayer(shape=(None, backprop_step), input_var=m)

    neural_network = input_layer;

    #batch_size, backprop_step, window_size = lasagne.layers.get_output_shape(neural_network)

    neural_network = lasagne.layers.EmbeddingLayer(neural_network,
                                                   input_size=vocabulary_dimension,
                                                   output_size=embedding_dimension,
                                                   W=lasagne.init.GlorotNormal());

    #self._embeddings = neural_network.get_params(trainable=True)[-1];
    #self._normalize_embeddings_function = theano.function(inputs=[],updates={self._embeddings: self._embeddings / theano.tensor.sqrt((self._embeddings ** 2).sum(axis=1)).dimshuffle(0, 'x')}        )

    neural_network = lasagne.layers.ReshapeLayer(neural_network, (-1, window_size * embedding_dimension));

    neural_network = lasagne.layers.ReshapeLayer(neural_network, (-1, backprop_step, lasagne.layers.get_output_shape(neural_network)[-1]));

    neural_network = BidirectionalRecurrentLayer(neural_network,
                                                 128,
                                                 forward_nonlinearity=nonlinearities.sigmoid,
                                                 backward_nonlinearity=nonlinearities.sigmoid,
                                                 mask_input=mask_layer,
                                                 W_forward_in_to_hid=lasagne.init.GlorotUniform(
                                                     gain=network.GlorotUniformGain[nonlinearities.sigmoid]),
                                                 W_forward_hid_to_hid=lasagne.init.GlorotUniform(
                                                     gain=network.GlorotUniformGain[nonlinearities.sigmoid]),
                                                 b_forward_in=lasagne.init.Constant(0.),
                                                 b_forward_hid=lasagne.init.Constant(0.),
                                                 W_backward_in_to_hid=lasagne.init.GlorotUniform(
                                                     gain=network.GlorotUniformGain[nonlinearities.sigmoid]),
                                                 W_backward_hid_to_hid=lasagne.init.GlorotUniform(
                                                     gain=network.GlorotUniformGain[nonlinearities.sigmoid]),
                                                 b_backward_in=lasagne.init.Constant(0.),
                                                 b_backward_hid=lasagne.init.Constant(0.),
                                                 )

    neural_network = lasagne.layers.DenseLayer(neural_network,
                                               128,
                                               W=lasagne.init.GlorotUniform(
                                                   gain=network.GlorotUniformGain[nonlinearities.softmax]),
                                               nonlinearity=nonlinearities.softmax)









    # Create a train_loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy train_loss):
    train_prediction = network.get_output()
    train_loss = network.get_objective_to_minimize(y);
    # train_loss = theano.tensor.mean(lasagne.objectives.categorical_crossentropy(train_prediction, y))
    train_accuracy = theano.tensor.mean(theano.tensor.eq(theano.tensor.argmax(train_prediction, axis=1), y),
                                        dtype=theano.config.floatX)

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    all_params = network.get_network_params(trainable=True)
    updates = lasagne.updates.nesterov_momentum(train_loss, all_params, 0.1, momentum=0.95)

    # Create a train_loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the networks,
    # disabling dropout layers.
    validate_prediction = network.get_output(deterministic=True)
    validate_loss = theano.tensor.mean(theano.tensor.nnet.categorical_crossentropy(validate_prediction, y))
    # As a bonus, also create an expression for the classification accuracy:
    validate_accuracy = theano.tensor.mean(theano.tensor.eq(theano.tensor.argmax(validate_prediction, axis=1), y),
                                           dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training train_loss:
    train_function = theano.function(
        inputs=[x, y, m],
        outputs=[train_loss, train_accuracy],
        updates=updates
    )

    # Compile a second function computing the validation train_loss and accuracy:
    validate_function = theano.function(
        inputs=[x, y, m],
        outputs=[validate_loss, validate_accuracy],
    )

    total_train_loss = 0;
    total_train_accuracy = 0;
    total_train_instances = 0;
    for train_sequence_x, train_sequence_y in zip(train_set_x, train_set_y):
        # context_windows = get_context_windows(train_sequence_x, window_size)
        # train_minibatch, train_minibatch_masks = get_mini_batches(context_windows, backprop_step);
        train_minibatch, train_minibatch_masks = network.get_mini_batches(train_sequence_x);
        assert len(train_minibatch) == len(train_minibatch_masks);
        assert len(train_minibatch) == len(train_sequence_y);
        # print mini_batches.shape, mini_batch_masks.shape, train_sequence_y.shape

        minibatch_average_train_loss, minibatch_average_train_accuracy = train_function(train_minibatch, train_sequence_y, train_minibatch_masks)

        # embedding_layer = [layer for layer in network.get_all_layers() if isinstance(layer, lasagne.layers.EmbeddingLayer)][0];
        # print numpy.sum(embedding_layer.W.eval()**2)

        # print numpy.sum(network._embeddings.eval()**2)
        # old_values = network._embeddings.eval()
        # normalize_embedding_function();
        network._normalize_embeddings_function();
        # print numpy.sum(network._embeddings.eval()**2)
        # new_values = network._embeddings.eval();

        total_train_loss += minibatch_average_train_loss * len(train_sequence_y);
        total_train_accuracy += minibatch_average_train_accuracy * len(train_sequence_y);
        total_train_instances += len(train_sequence_y);




















def get_mini_batches(self, sequence):
    '''
    context_windows :: list of word idxs
    return a list of minibatches of indexes
    which size is equal to backprop_step
    border cases are treated as follow:
    eg: [0,1,2,3] and backprop_step = 3
    will output:
    [[0],[0,1],[0,1,2],[1,2,3]]
    '''

    context_windows = get_context_windows(sequence, self._window_size);
    mini_batches, mini_batch_masks = get_mini_batches(context_windows, self._backprop_step);
    return mini_batches, mini_batch_masks


'''
embedding_layer = lasagne.layers.EmbeddingLayer(input_layer,
                                                input_size=vocabulary_dimension,
                                                output_size=embedding_dimension,
                                                W=lasagne.init.GlorotUniform());
print "----------", lasagne.layers.get_output_shape(embedding_layer, (10, 46))
'''

import rnn

network = rnn.RecurrentNeuralNetwork(
    input_network=input_layer,
    input_mask=mask_layer,
    vocabulary_dimension=vocabulary_dimension,
    embedding_dimension=embedding_dimension,
    layer_dimensions=(pre_rnn_layer_dimensions, rnn_layer_dimensions, post_rnn_layer_dimensions),
    layer_nonlinearities=(pre_rnn_layer_nonlinearities, rnn_layer_nonlinearities, post_rnn_layer_nonlinearities),
    objective_to_minimize=objective_to_minimize,
)
