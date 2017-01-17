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
import network

from layers.dropout import GeneralizedDropoutLayer, sample_activation_probability

class ElmanNetwork(network.Network):
    def __init__(self,
                 input_network=None,
                 input_mask=None,

                 vocabulary_dimension=None,
                 embedding_dimension=None,

                 window_size=-1,
                 position_offset=-1,
                 sequence_length=-1,

                 layer_dimensions=None,
                 layer_nonlinearities=None,

                 dense_activation_parameters=None,
                 dense_activation_styles=None,

                 recurrent_type="RecurrentLayer",

                 # layer_activation_parameters=None,
                 # layer_activation_styles=None,
                 objective_to_minimize=None,
                 # pretrained_model=None,
                 ):
        super(ElmanNetwork, self).__init__(input_network)
        assert window_size > 0;
        assert sequence_length > 0;

        #
        #
        #
        #
        #

        dropout_layer_index = 0;

        #
        #
        #
        #
        #

        self._input_data_layer = input_network;
        self._input_mask_layer = input_mask;

        neural_network = input_network;

        batch_size, input_sequence_length, input_window_size = lasagne.layers.get_output_shape(neural_network)
        assert sequence_length == input_sequence_length;
        assert window_size == input_window_size
        self._window_size = window_size;
        self._position_offset = position_offset;
        self._sequence_length = sequence_length;

        neural_network = lasagne.layers.EmbeddingLayer(neural_network,
                                                       input_size=vocabulary_dimension,
                                                       output_size=embedding_dimension,
                                                       W=lasagne.init.GlorotNormal());
        print_output_dimension("after embedding layer", neural_network, batch_size, sequence_length, window_size);

        self._embeddings = neural_network.get_params(trainable=True)[-1];
        self._normalize_embeddings_function = theano.function(
            inputs=[],
            updates={self._embeddings: self._embeddings / theano.tensor.sqrt((self._embeddings ** 2).sum(axis=1)).dimshuffle(0, 'x')}
        )

        neural_network = lasagne.layers.ReshapeLayer(neural_network, (-1, self._sequence_length, self._window_size * embedding_dimension));
        print_output_dimension("after window merge", neural_network, batch_size, sequence_length, window_size);

        #
        #
        #
        #
        #

        input_layer_shape = lasagne.layers.get_output_shape(neural_network)[1:]
        previous_layer_shape = numpy.prod(input_layer_shape)

        activation_probability = sample_activation_probability(previous_layer_shape,
                                                               dense_activation_styles[dropout_layer_index],
                                                               dense_activation_parameters[
                                                                   dropout_layer_index]);
        activation_probability = numpy.reshape(activation_probability, input_layer_shape)
        dropout_layer_index += 1;

        neural_network = GeneralizedDropoutLayer(neural_network, activation_probability=activation_probability);

        #
        #
        #
        #
        #

        last_rnn_layer_index = 0;
        for layer_index in xrange(len(layer_dimensions)):
            layer_dimension = layer_dimensions[layer_index]
            if isinstance(layer_dimension, list):
                last_rnn_layer_index = layer_index;

        recurrent_layer = getattr(lasagne.layers.recurrent, recurrent_type)

        for layer_index in xrange(len(layer_dimensions)):
            layer_dimension = layer_dimensions[layer_index]
            layer_nonlinearity = layer_nonlinearities[layer_index]

            if isinstance(layer_dimension, int):
                if layer_index <= last_rnn_layer_index:
                    neural_network = lasagne.layers.ReshapeLayer(neural_network, (-1, lasagne.layers.get_output_shape(neural_network)[-1]));
                    print_output_dimension("after reshape (for dense layer)", neural_network, batch_size, sequence_length, window_size);

                neural_network = lasagne.layers.DenseLayer(neural_network,
                                                           layer_dimension,
                                                           W=lasagne.init.GlorotUniform(
                                                               gain=network.GlorotUniformGain[
                                                                   layer_nonlinearity]),
                                                           nonlinearity=layer_nonlinearity)
                print_output_dimension("after dense layer %i" % layer_index, neural_network, batch_size, sequence_length, window_size);

                #
                #
                #
                #
                #

                if dropout_layer_index >= len(dense_activation_styles):
                    continue;

                input_layer_shape = lasagne.layers.get_output_shape(neural_network)[1:]
                previous_layer_shape = numpy.prod(input_layer_shape)

                activation_probability = sample_activation_probability(previous_layer_shape,
                                                                       dense_activation_styles[dropout_layer_index],
                                                                       dense_activation_parameters[
                                                                           dropout_layer_index]);
                activation_probability = numpy.reshape(activation_probability, input_layer_shape)
                dropout_layer_index += 1;

                neural_network = GeneralizedDropoutLayer(neural_network, activation_probability=activation_probability);

                #
                #
                #
                #
                #

            elif isinstance(layer_dimension, list):
                assert isinstance(layer_nonlinearity, list)
                if not isinstance(lasagne.layers.get_all_layers(neural_network)[-1], recurrent_layer):
                    neural_network = lasagne.layers.ReshapeLayer(neural_network, (-1, self._sequence_length, lasagne.layers.get_output_shape(neural_network)[-1]));
                    print_output_dimension("after reshape (for recurrent layer)", neural_network, batch_size, sequence_length, window_size);

                layer_dimension = layer_dimension[0]
                layer_nonlinearity = layer_nonlinearity[0]
                #neural_network = lasagne.layers.RecurrentLayer(neural_network,
                if recurrent_layer==lasagne.layers.recurrent.RecurrentLayer:
                    neural_network = lasagne.layers.RecurrentLayer(neural_network,
                                                                   layer_dimension,
                                                                   W_in_to_hid=lasagne.init.GlorotUniform(
                                                                       gain=network.GlorotUniformGain[
                                                                           layer_nonlinearity]),
                                                                   W_hid_to_hid=lasagne.init.GlorotUniform(
                                                                       gain=network.GlorotUniformGain[
                                                                           layer_nonlinearity]),
                                                                   b=lasagne.init.Constant(0.),
                                                                   nonlinearity=layer_nonlinearity,
                                                                   hid_init=lasagne.init.Constant(0.),
                                                                   backwards=False,
                                                                   learn_init=False,
                                                                   gradient_steps=-1,
                                                                   grad_clipping=0,
                                                                   unroll_scan=False,
                                                                   precompute_input=True,
                                                                   mask_input=input_mask,
                                                                   # only_return_final=True
                                                                   );
                elif recurrent_layer==lasagne.layers.recurrent.LSTMLayer:
                    neural_network = lasagne.layers.LSTMLayer(neural_network,
                                                              layer_dimension,
                                                              ingate=lasagne.layers.Gate(),
                                                              forgetgate=lasagne.layers.Gate(),
                                                              cell=lasagne.layers.Gate(W_cell=None, nonlinearity=layer_nonlinearity),
                                                              outgate=lasagne.layers.Gate(),
                                                              nonlinearity=layer_nonlinearity,
                                                              cell_init=lasagne.init.Constant(0.),
                                                              hid_init=lasagne.init.Constant(0.),
                                                              backwards=False,
                                                              learn_init=False,
                                                              peepholes=True,
                                                              gradient_steps=-1,
                                                              grad_clipping=0,
                                                              unroll_scan=False,
                                                              precompute_input=True,
                                                              mask_input=input_mask,
                                                              # only_return_final=True
                                                              );
                print_output_dimension("after recurrent layer %i" % layer_index, neural_network, batch_size, sequence_length, window_size);
            else:
                sys.stderr.write("layer specification conflicts...\n")
                sys.exit();

        self._neural_network = neural_network;

        assert objective_to_minimize != None;
        self._objective_to_minimize = objective_to_minimize;

    '''
    def get_output_from_sequence(self, input_sequence):
        context_windows = get_context_windows(input_sequence, self._window_size)
        train_minibatch, train_minibatch_masks = get_mini_batches(context_windows, self._backprop_step);
        # lasagne.layers.get_output(self._neural_network, inputs, **kwargs)
        return self.get_output({self._input_data_layer:train_minibatch, self._input_mask_layer:train_minibatch_masks});
    '''

    '''
    def get_objective_to_minimize(self, label):
        train_loss = theano.tensor.mean(self._objective_to_minimize(self.get_output(), label))

        train_loss += self.L1_regularizer();
        train_loss += self.L2_regularizer();
        #train_loss += self.dae_regularizer();

        return train_loss
    '''

    def get_instance_sequences(self, instance):
        '''
        context_windows :: list of word idxs
        return a list of minibatches of indexes
        which size is equal to backprop_step
        border cases are treated as follow:
        eg: [0,1,2,3] and backprop_step = 3
        will output:
        [[0],[0,1],[0,1,2],[1,2,3]]
        '''

        context_windows = get_context_windows(instance, self._window_size, self._position_offset);
        sequences_x, sequences_m = get_sequences(context_windows, self._sequence_length);
        return sequences_x, sequences_m

def get_context_windows(instance, window_size, position_offset=-1, vocab_size=None):
    '''
    window_size :: int corresponding to the size of the window
    given a list of indexes composing a sentence
    it will return a list of list of indexes corresponding
    to context windows surrounding each word in the sentence
    '''

    assert window_size >= 1
    if position_offset < 0:
        assert window_size % 2 == 1
        position_offset = window_size / 2;
    assert position_offset < window_size

    instance = list(instance)

    if vocab_size == None:
        context_windows = -numpy.ones((len(instance), window_size), dtype=numpy.int32);
        #padded_sequence = window_size / 2 * [-1] + instance + window_size / 2 * [-1]
        padded_sequence = position_offset * [-1] + instance + (window_size - position_offset) * [-1]
        for i in xrange(len(instance)):
            context_windows[i, :] = padded_sequence[i:i + window_size];
    else:
        context_windows = numpy.zeros((len(instance), vocab_size), dtype=numpy.int32);
        #padded_sequence = window_size / 2 * [-1] + instance + window_size / 2 * [-1]
        padded_sequence = position_offset * [-1] + instance + (window_size - position_offset) * [-1]
        for i in xrange(len(instance)):
            for j in padded_sequence[i:i + window_size]:
                context_windows[i, j] += 1;

    # assert len(context_windows) == len(sequence)
    return context_windows

def get_sequences(context_windows, sequence_length):
    '''
    context_windows :: list of word idxs
    return a list of minibatches of indexes
    which size is equal to backprop_step
    border cases are treated as follow:
    eg: [0,1,2,3] and backprop_step = 3
    will output:
    [[0],[0,1],[0,1,2],[1,2,3]]
    '''

    number_of_tokens, window_size = context_windows.shape;
    sequences_x = -numpy.ones((number_of_tokens, sequence_length, window_size), dtype=numpy.int32);
    sequences_m = numpy.zeros((number_of_tokens, sequence_length), dtype=numpy.int32);
    for i in xrange(min(number_of_tokens, sequence_length)):
        sequences_x[i, 0:i + 1, :] = context_windows[0:i + 1, :];
        sequences_m[i, 0:i + 1] = 1;
    for i in xrange(min(number_of_tokens, sequence_length), number_of_tokens):
        sequences_x[i, :, :] = context_windows[i - sequence_length + 1:i + 1, :];
        sequences_m[i, :] = 1;
    return sequences_x, sequences_m

def print_output_dimension(checkpoint_text, neural_network, batch_size, sequence_length, window_size):
    reference_to_input_layers = [input_layer for input_layer in lasagne.layers.get_all_layers(neural_network) if
                                 isinstance(input_layer, lasagne.layers.InputLayer)];
    if len(reference_to_input_layers) == 1:
        print checkpoint_text, ":", lasagne.layers.get_output_shape(neural_network, {
            reference_to_input_layers[0]: (batch_size, sequence_length, window_size)})
    elif len(reference_to_input_layers) == 2:
        print checkpoint_text, ":", lasagne.layers.get_output_shape(neural_network, {
            reference_to_input_layers[0]: (batch_size, sequence_length, window_size),
            reference_to_input_layers[1]: (batch_size, sequence_length)})

if __name__ == '__main__':
    window_size = 5;
    position_offset = 1;
    sequence_length = 9;

    network = ElmanNetwork(
        input_network=lasagne.layers.InputLayer(shape=(None, sequence_length, window_size,)),
        input_mask=lasagne.layers.InputLayer(shape=(None, sequence_length)),
        vocabulary_dimension=100,
        embedding_dimension=50,
        window_size=window_size,
        position_offset=position_offset,
        sequence_length=sequence_length,
        layer_dimensions=[32, [64], 127],
        layer_nonlinearities=[lasagne.nonlinearities.rectify, [lasagne.nonlinearities.rectify], lasagne.nonlinearities.softmax],
        objective_to_minimize=lasagne.objectives.categorical_crossentropy,
    )

    data = [554, 23, 241, 534, 358, 136, 193, 11, 208, 251, 104, 502, 413, 256, 104];
    context_windows = get_context_windows(data, window_size);
    print context_windows;
    mini_batches, mini_batch_masks = network.get_instance_sequences(data);
    print mini_batches;
    print mini_batch_masks;
