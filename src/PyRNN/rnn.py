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

backprop_step_example=9
window_size_example=5
batch_size_example=13

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
                 ):
        super(RecurrentNeuralNetwork, self).__init__(input_network)

        pre_rnn_layer_dimensions, rnn_layer_dimensions, post_rnn_layer_dimensions = layer_dimensions
        pre_rnn_layer_nonlinearities, rnn_layer_nonlinearities, post_rnn_layer_nonlinearities = layer_nonlinearities;
        # assert len(rnn_layer_dimensions) == len(rnn_layer_nonlinearities)
        # assert len(rnn_layer_dimensions) == len(layer_activation_parameters)
        # assert len(rnn_layer_dimensions) == len(layer_activation_styles) 

        neural_network = input_network;

        batch_size, backprop_step, window_size = lasagne.layers.get_output_shape(neural_network)
        #print batch_size, backprop_step, window_size

        # '''
        neural_network = lasagne.layers.EmbeddingLayer(neural_network,
                                                       input_size=vocabulary_dimension,
                                                       output_size=embedding_dimension,
                                                       W=lasagne.init.GlorotUniform());

        self._embedding = neural_network.get_params(trainable=True)[-1];
        print type(self._embedding);

        '''
        print self._embedding.eval()
        self._normalize_embedding_function = theano.function(
            inputs=[],
            updates={self._embedding: self._embedding / theano.tensor.sqrt((self._embedding ** 2).sum(axis=1))}
        )
        print self._embedding.eval()
        '''

        #print "checkpoint a", lasagne.layers.get_output_shape(neural_network, (batch_size_example, backprop_step_example, window_size_example))

        neural_network = lasagne.layers.ReshapeLayer(neural_network, (-1, backprop_step, window_size * embedding_dimension));
        #print "checkpoint a", lasagne.layers.get_output_shape(neural_network, (batch_size_example, backprop_step_example, window_size_example))

        for pre_rnn_layer_index in xrange(len(pre_rnn_layer_dimensions)):
            # previous_layer_dimension = lasagne.layers.get_output_shape(neural_network)[1:];
            # activation_probability = sample_activation_probability(previous_layer_dimension, layer_activation_styles[pre_rnn_layer_index], layer_activation_parameters[pre_rnn_layer_index]);

            # neural_network = GeneralizedDropoutLayer(neural_network, activation_probability=activation_probability);

            pre_rnn_layer_dimension = pre_rnn_layer_dimensions[pre_rnn_layer_index]
            pre_rnn_layer_nonlinearity = pre_rnn_layer_nonlinearities[pre_rnn_layer_index];

            neural_network = lasagne.layers.DenseLayer(neural_network,
                                                       pre_rnn_layer_dimension,
                                                       W=lasagne.init.GlorotUniform(gain=network.GlorotUniformGain[pre_rnn_layer_nonlinearity]),
                                                       nonlinearity=pre_rnn_layer_nonlinearity)

            #print "checkpoint b", pre_rnn_layer_index, lasagne.layers.get_output_shape(neural_network, (batch_size_example, backprop_step_example, window_size_example))

        neural_network = lasagne.layers.ReshapeLayer(neural_network, (-1, backprop_step, window_size, lasagne.layers.get_output_shape(neural_network)[-1] / window_size));
        #print "checkpoint b", lasagne.layers.get_output_shape(neural_network, (batch_size_example, backprop_step_example, window_size_example))

        for rnn_layer_index in xrange(len(rnn_layer_dimensions)):
            # previous_layer_dimension = lasagne.layers.get_output_shape(neural_network)[1:];
            # activation_probability = sample_activation_probability(previous_layer_dimension, layer_activation_styles[rnn_layer_index], layer_activation_parameters[rnn_layer_index]);

            # neural_network = GeneralizedDropoutLayer(neural_network, activation_probability=activation_probability);

            rnn_layer_dimension = rnn_layer_dimensions[rnn_layer_index]
            rnn_layer_nonlinearity = rnn_layer_nonlinearities[rnn_layer_index];

            neural_network = lasagne.layers.RecurrentLayer(neural_network,
                                                           rnn_layer_dimension,
                                                           W_in_to_hid=lasagne.init.GlorotUniform(gain=network.GlorotUniformGain[rnn_layer_nonlinearity]),
                                                           W_hid_to_hid=lasagne.init.GlorotUniform(gain=network.GlorotUniformGain[rnn_layer_nonlinearity]),
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

    '''
    def normalize_embeddings(self):
        print self._embedding.eval()
        self._normalize_embedding_function();
        print self._embedding.eval()
    '''