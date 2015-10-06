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
import dae

import network

class MultiLayerPerceptron2(network.Network):
    def __init__(self,
            input=None,
            layer_nonlinearities=None,
            layer_shapes=None,
            # layer_dropout_rates=None,
            # objective_to_minimize=network.mean_categorical_crossentropy,
            # L1_regularizer_lambdas=None,
            # L2_regularizer_lambdas=None
            ):
        super(MultiLayerPerceptron2, self).__init__(
            # objective_to_minimize,
            )
        
        # if layer_dropout_rates is not None:
            # layer_dropout_rates = [0] * (len(layer_shapes)-1);

        network = lasagne.layers.InputLayer(shape=(None, layer_shapes[0]), input_var=input)
        # print _network.shape, _network.output_shape
        for layer_index in xrange(1, len(layer_shapes)):
            # network = lasagne.layers.DropoutLayer(network, p=layer_dropout_rates[layer_index - 1])
            
            layer_shape = layer_shapes[layer_index]
            layer_nonlinearity = layer_nonlinearities[layer_index - 1];
            network = lasagne.layers.DenseLayer(network, layer_shape, nonlinearity=layer_nonlinearity)
        
        self._network = network;
        
        # self.set_L1_regularizer_lambda(L1_regularizer_lambdas);
        # self.set_L2_regularizer_lambda(L2_regularizer_lambdas);
        
    def get_pretrain_daes(self, layer_corruption_levels):
        layers = self.get_all_layers();
        
        denoising_auto_encoders = [];
        
        assert len(layer_corruption_levels) == len(layers) - 2;
        for hidden_layer_index in xrange(1, len(layers) - 1):
            hidden_layer = layers[hidden_layer_index];
            hidden_layer_shape = hidden_layer.num_units;
            hidden_layer_nonlinearity = hidden_layer.nonlinearity
            
            layer_corruption_level = layer_corruption_levels[hidden_layer_index - 1];
            
            denoising_auto_encoder = dae.DenoisingAutoEncoderLayer(
                hidden_layer.input_layer,
                num_units=hidden_layer_shape,
                corruption_level=layer_corruption_level,
                W_encode=hidden_layer.W,
                b_encoder=hidden_layer.b,
                encoder_nonlinearity=hidden_layer_nonlinearity,
                decoder_nonlinearity=hidden_layer_nonlinearity,
                )
            
            denoising_auto_encoders.append(denoising_auto_encoder);

        return denoising_auto_encoders;
