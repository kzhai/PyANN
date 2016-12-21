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

backprop_step_example = 9
window_size_example = 5
batch_size_example = 13

class LongShortTermMemory(network.Network):
    def __init__(self,
                 input_network=None,
                 input_mask=None,

                 vocabulary_dimension=None,
                 embedding_dimension=None,

                 layer_dimensions=None,
                 layer_nonlinearities=None,

                 # layer_activation_parameters=None,
                 # layer_activation_styles=None,
                 objective_to_minimize=None,
                 # pretrained_model=None,

                 #window_size=5,
                 #backprop_step=9,
                 ):
        super(RecurrentNeuralNetwork, self).__init__(input_network)

        self._input_data_layer = input_network;
        self._input_mask_layer = input_mask;

        neural_network = input_network;

        batch_size, backprop_step, window_size = lasagne.layers.get_output_shape(neural_network)
        self._window_size = window_size;
        self._backprop_step = backprop_step;

        #print "checkpoint a", lasagne.layers.get_output_shape(neural_network, (batch_size_example, backprop_step_example, window_size_example))
        # (13, 9, 5)

        neural_network = lasagne.layers.EmbeddingLayer(neural_network,
                                                       input_size=vocabulary_dimension,
                                                       output_size=embedding_dimension,
                                                       W=lasagne.init.GlorotNormal());

        self._embeddings = neural_network.get_params(trainable=True)[-1];
        self._normalize_embeddings_function = theano.function(
            inputs=[],
            updates={self._embeddings: self._embeddings / theano.tensor.sqrt((self._embeddings ** 2).sum(axis=1)).dimshuffle(0, 'x')}
        )

        neural_network = lasagne.layers.ReshapeLayer(neural_network, (-1, backprop_step, self._window_size * embedding_dimension));

        last_lstm_layer_index = 0;
        for layer_index in xrange(len(layer_dimensions)):
            layer_dimension = layer_dimensions[layer_index]
            if isinstance(layer_dimension, list):
                last_lstm_layer_index = layer_index;

        def print_output_dimension(checkpoint_text=""):
            reference_to_input_layers = [input_layer for input_layer in lasagne.layers.get_all_layers(neural_network) if
                                         isinstance(input_layer, lasagne.layers.InputLayer)];
            if len(reference_to_input_layers) == 1:
                print checkpoint_text, ":", "layer", layer_index, lasagne.layers.get_output_shape(neural_network, {
                    reference_to_input_layers[0]: (batch_size_example, backprop_step_example, window_size_example)})
            elif len(reference_to_input_layers) == 2:
                print checkpoint_text, ":", "layer", layer_index, lasagne.layers.get_output_shape(neural_network, {
                    reference_to_input_layers[0]: (batch_size_example, backprop_step_example, window_size_example),
                    reference_to_input_layers[1]: (batch_size_example, backprop_step_example)})

        for layer_index in xrange(len(layer_dimensions)):
            layer_dimension = layer_dimensions[layer_index]
            layer_nonlinearity = layer_nonlinearities[layer_index]

            if isinstance(layer_dimension, int):
                if layer_index <= last_lstm_layer_index:
                    neural_network = lasagne.layers.ReshapeLayer(neural_network, (-1, lasagne.layers.get_output_shape(neural_network)[-1]));
                    #print_output_dimension("checkpoint a1");

                neural_network = lasagne.layers.DenseLayer(neural_network,
                                                           layer_dimension,
                                                           W=lasagne.init.GlorotUniform(
                                                               gain=network.GlorotUniformGain[
                                                                   layer_nonlinearity]),
                                                           nonlinearity=layer_nonlinearity)
                #print_output_dimension("checkpoint a2");
            elif isinstance(layer_dimension, list):
                assert isinstance(layer_nonlinearity, list)
                if not isinstance(lasagne.layers.get_all_layers(neural_network)[-1], lasagne.layers.RecurrentLayer):
                    neural_network = lasagne.layers.ReshapeLayer(neural_network, (-1, backprop_step, lasagne.layers.get_output_shape(neural_network)[-1]));
                    #print_output_dimension("checkpoint b1");

                layer_dimension = layer_dimension[0]
                layer_nonlinearity = layer_nonlinearity[0]
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
                #print_output_dimension("checkpoint b2");
            else:
                sys.stderr.write("layer specification conflicts...\n")
                sys.exit();

        #print lasagne.layers.get_all_layers(neural_network);

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

if __name__ == '__main__':
    window_size = 5;
    backprop_step = 9;

    network = RecurrentNeuralNetwork(
        input_network=lasagne.layers.InputLayer(shape=(None, backprop_step, window_size,)),
        input_mask=lasagne.layers.InputLayer(shape=(None, backprop_step)),
        vocabulary_dimension=100,
        embedding_dimension=50,
        layer_dimensions=[64, [128], 10],
        layer_nonlinearities=[lasagne.nonlinearities.rectify, [lasagne.nonlinearities.rectify], lasagne.nonlinearities.softmax],
        objective_to_minimize=lasagne.objectives.categorical_crossentropy,
    )

    data = [554, 23, 241, 534, 358, 136, 193, 11, 208, 251, 104, 502, 413, 256, 104];
    #context_windows = network.get_context_windows(data);
    #print context_windows;
    mini_batches, mini_batch_masks = network.get_mini_batches(data);
    print mini_batches;
    print mini_batch_masks;
