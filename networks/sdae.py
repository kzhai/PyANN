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

from networks.dae import DenoisingAutoEncoder

class StackedDenoisingAutoEncoder():
    def __init__(self,
            input=None,
            layer_shapes=None,
            layer_nonlinearities=None,
            
            layer_corruption_levels=None,
            
            L1_regularizer_lambdas=None,
            L2_regularizer_lambdas=None,
            
            objective_to_minimize=lasagne.objectives.binary_crossentropy,
            ):
        assert len(layer_shapes) == len(layer_nonlinearities) + 1
        # assert len(layer_activation_parameters) == len(layer_nonlinearities)
        # assert len(layer_activation_styles) == len(layer_nonlinearities)
        
        network = lasagne.layers.InputLayer(shape=(None, layer_shapes[0]), input_var=input)
        for layer_index in xrange(1, len(layer_shapes)):
            layer_shape = layer_shapes[layer_index]
            layer_nonlinearity = layer_nonlinearities[layer_index - 1];
            network = lasagne.layers.DenseLayer(network, layer_shape, nonlinearity=layer_nonlinearity)
        self.network = network;
        
        denoising_auto_encoders = [];
        layers = lasagne.layers.get_all_layers(network)
        if layer_corruption_levels is None:
            layer_corruption_levels = numpy.zeros(len(layers) - 1)
        assert len(layer_corruption_levels) == len(layers) - 1;
        
        for hidden_layer_index in xrange(1, len(layers)):
            hidden_layer = layers[hidden_layer_index];
            # this is to get around the dropout layer
            input_layer = layers[hidden_layer_index - 1];
            hidden_layer_shape = hidden_layer.num_units;
            hidden_layer_nonlinearity = hidden_layer.nonlinearity
            
            # this is to get around the dropout layer
            layer_corruption_level = layer_corruption_levels[hidden_layer_index - 1];
            
            denoising_auto_encoder = DenoisingAutoEncoder(
                input=input_layer,
                layer_shape=hidden_layer_shape,
                encoder_nonlinearity=hidden_layer_nonlinearity,
                #decoder_nonlinearity=lasagne.nonlinearities.identity,
                decoder_nonlinearity=lasagne.nonlinearities.sigmoid,
                objective_to_minimize=objective_to_minimize,
                # objective_to_minimize=lasagne.objectives.binary_crossentropy,
                corruption_level=layer_corruption_level,
                
                W_encode=hidden_layer.W,
                b_encoder=hidden_layer.b,
                )
            
            denoising_auto_encoder.set_L1_regularizer_lambda(L1_regularizer_lambdas)
            denoising_auto_encoder.set_L2_regularizer_lambda(L2_regularizer_lambdas)
            
            denoising_auto_encoders.append(denoising_auto_encoder);
            
        self.denoising_auto_encoders = denoising_auto_encoders;