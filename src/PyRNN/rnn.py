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

class RecurrentNeuralNetwork(network.Network):
    def __init__(self,
                 input_network=None,
                 input_mask=None,

                 vocabulary_dimension=None,
                 embedding_dimension=None,

                 layer_dimensions=None,
                 layer_nonlinearities=None,

                 # rnn_layer_dimensions=None,
                 # rnn_layer_nonlinearities=None,

                 # post_rnn_layer_dimensions=None,
                 # post_rnn_layer_nonlinearities=None,

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
                                                       W=lasagne.init.GlorotUniform());

        self._embedding = neural_network.get_params(trainable=True)[-1];
        #print self._embedding.eval()
        self._normalize_embedding_function = theano.function(
            inputs=[],
            updates={self._embedding: self._embedding / theano.tensor.sqrt((self._embedding ** 2).sum(axis=1))}
        )
        #print self._embedding.eval()

        #print "checkpoint a", lasagne.layers.get_output_shape(neural_network, (batch_size_example, backprop_step_example, window_size_example))
        #(13, 9, 5, 100)

        neural_network = lasagne.layers.ReshapeLayer(neural_network, (-1, self._window_size * embedding_dimension));
        #print "checkpoint a", lasagne.layers.get_output_shape(neural_network, (batch_size_example, backprop_step_example, window_size_example))
        #(117, 5, 100)

        pre_rnn_layer_dimensions, rnn_layer_dimensions, post_rnn_layer_dimensions = layer_dimensions
        pre_rnn_layer_nonlinearities, rnn_layer_nonlinearities, post_rnn_layer_nonlinearities = layer_nonlinearities;
        # assert len(rnn_layer_dimensions) == len(rnn_layer_nonlinearities)
        # assert len(rnn_layer_dimensions) == len(layer_activation_parameters)
        # assert len(rnn_layer_dimensions) == len(layer_activation_styles)

        for pre_rnn_layer_index in xrange(len(pre_rnn_layer_dimensions)):
            # previous_layer_dimension = lasagne.layers.get_output_shape(neural_network)[1:];
            # activation_probability = sample_activation_probability(previous_layer_dimension, layer_activation_styles[pre_rnn_layer_index], layer_activation_parameters[pre_rnn_layer_index]);

            # neural_network = GeneralizedDropoutLayer(neural_network, activation_probability=activation_probability);

            pre_rnn_layer_dimension = pre_rnn_layer_dimensions[pre_rnn_layer_index]
            pre_rnn_layer_nonlinearity = pre_rnn_layer_nonlinearities[pre_rnn_layer_index];

            neural_network = lasagne.layers.DenseLayer(neural_network,
                                                       pre_rnn_layer_dimension,
                                                       W=lasagne.init.GlorotUniform(
                                                           gain=network.GlorotUniformGain[pre_rnn_layer_nonlinearity]),
                                                       nonlinearity=pre_rnn_layer_nonlinearity)

            #print "checkpoint b", pre_rnn_layer_index, lasagne.layers.get_output_shape(neural_network, (batch_size_example, backprop_step_example, window_size_example))
            # (117, 256)

        neural_network = lasagne.layers.ReshapeLayer(neural_network, (-1, backprop_step, lasagne.layers.get_output_shape(neural_network)[-1]));
        #print "checkpoint b", lasagne.layers.get_output_shape(neural_network, (batch_size_example, backprop_step_example, window_size_example))
        # (13, 9, 256)

        for rnn_layer_index in xrange(len(rnn_layer_dimensions)):
            # previous_layer_dimension = lasagne.layers.get_output_shape(neural_network)[1:];
            # activation_probability = sample_activation_probability(previous_layer_dimension, layer_activation_styles[rnn_layer_index], layer_activation_parameters[rnn_layer_index]);

            # neural_network = GeneralizedDropoutLayer(neural_network, activation_probability=activation_probability);

            rnn_layer_dimension = rnn_layer_dimensions[rnn_layer_index]
            rnn_layer_nonlinearity = rnn_layer_nonlinearities[rnn_layer_index];

            neural_network = lasagne.layers.RecurrentLayer(neural_network,
                                                           rnn_layer_dimension,
                                                           W_in_to_hid=lasagne.init.GlorotUniform(
                                                               gain=network.GlorotUniformGain[rnn_layer_nonlinearity]),
                                                           W_hid_to_hid=lasagne.init.GlorotUniform(
                                                               gain=network.GlorotUniformGain[rnn_layer_nonlinearity]),
                                                           b=lasagne.init.Constant(0.),
                                                           nonlinearity=rnn_layer_nonlinearity,
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

            '''
            x = lasagne.layers.get_all_layers(neural_network)[0];
            m = lasagne.layers.get_all_layers(neural_network)[2];
            print "checkpoint c", rnn_layer_index, lasagne.layers.get_output_shape(neural_network, {x:(batch_size_example, backprop_step_example, window_size_example), m:(batch_size_example, backprop_step_example)})
            '''

        for post_rnn_layer_index in xrange(len(post_rnn_layer_dimensions)):
            # previous_layer_dimension = lasagne.layers.get_output_shape(neural_network)[1:];
            # activation_probability = sample_activation_probability(previous_layer_dimension, layer_activation_styles[post_rnn_layer_index], layer_activation_parameters[post_rnn_layer_index]);

            # neural_network = GeneralizedDropoutLayer(neural_network, activation_probability=activation_probability);

            post_rnn_layer_dimension = post_rnn_layer_dimensions[post_rnn_layer_index]
            post_rnn_layer_nonlinearity = post_rnn_layer_nonlinearities[post_rnn_layer_index];

            neural_network = lasagne.layers.DenseLayer(neural_network,
                                                       post_rnn_layer_dimension,
                                                       W=lasagne.init.GlorotUniform(
                                                           gain=network.GlorotUniformGain[post_rnn_layer_nonlinearity]),
                                                       nonlinearity=post_rnn_layer_nonlinearity)

            '''
            x = lasagne.layers.get_all_layers(neural_network)[0];
            m = lasagne.layers.get_all_layers(neural_network)[2];
            print "checkpoint d", post_rnn_layer_index, lasagne.layers.get_output_shape(neural_network, {x:(batch_size_example, backprop_step_example, window_size_example), m:(batch_size_example, backprop_step_example)})
            '''

        self._neural_network = neural_network;

        assert objective_to_minimize != None;
        self._objective_to_minimize = objective_to_minimize;

    def get_output_from_sequence(self, input_sequence):
        context_windows = get_context_windows(input_sequence, self._window_size)
        train_minibatch, train_minibatch_masks = get_mini_batches(context_windows, self._backprop_step);
        # lasagne.layers.get_output(self._neural_network, inputs, **kwargs)
        return self.get_output({self._input_data_layer:train_minibatch, self._input_mask_layer:train_minibatch_masks});

    '''
    def get_objective_to_minimize(self, label):
        train_loss = theano.tensor.mean(self._objective_to_minimize(self.get_output(), label))

        train_loss += self.L1_regularizer();
        train_loss += self.L2_regularizer();
        #train_loss += self.dae_regularizer();

        return train_loss
    '''

    '''
    normalize_function = theano.function(
        inputs = [],
        updates = {self.emb:self.emb/T.sqrt((self.emb**2).sum(axis=1)).dimshuffle(0,'x')}
    )

    average_train_loss, average_train_accuracy = train_function(mini_batches, train_sequence_y, mini_batch_masks)
    '''

    def normalize_embeddings(self):
        print self._embedding.eval()
        self._normalize_embedding_function();
        print self._embedding.eval()

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

if __name__ == '__main__':
    data = [554, 23, 241, 534, 358, 136, 193, 11, 208, 251, 104, 502, 413, 256, 104];
    #data = data[:4];
    window_size = 5;
    backprop_step = 9;
    a = get_context_windows(data, window_size);
    mini_batches, mini_batch_masks = get_mini_batches(a, backprop_step);
    print a;
    print mini_batches;
    print mini_batch_masks;