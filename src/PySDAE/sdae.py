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

#from layers.dae import DenoisingAutoEncoderLayer
from PyDAE.dae import DenoisingAutoEncoder

from theano.tensor.shared_randomstreams import RandomStreams

class StackedDenoisingAutoEncoder(network.Network):
    def __init__(self,
            input_network=None,
            layer_shapes=None,
            layer_nonlinearities=None,
            
            layer_corruption_levels=None,
            
            L1_regularizer_lambdas=None,
            L2_regularizer_lambdas=None,
            
            objective_to_minimize=lasagne.objectives.binary_crossentropy,
            ):
        self.input = lasagne.layers.get_output(input_network);

        assert len(layer_shapes) == len(layer_nonlinearities)
        assert len(layer_shapes) == len(layer_corruption_levels)
        # assert len(layer_activation_parameters) == len(layer_nonlinearities)
        # assert len(layer_activation_styles) == len(layer_nonlinearities)
        assert len(layer_shapes) == len(L1_regularizer_lambdas)
        assert len(layer_shapes) == len(L2_regularizer_lambdas)
        
        network = input_network;
        denoising_auto_encoders = [];
        for layer_index in xrange(len(layer_shapes)):
            layer_shape = layer_shapes[layer_index]
            layer_nonlinearity = layer_nonlinearities[layer_index];
            
            input_layer = network;
            network = lasagne.layers.DenseLayer(network, layer_shape, nonlinearity=layer_nonlinearity)
            hidden_layer = network;
            
            layer_corruption_level = layer_corruption_levels[layer_index];
            denoising_auto_encoder = DenoisingAutoEncoder(
                input_layer,
                layer_shape,
                encoder_nonlinearity=layer_nonlinearity,
                # decoder_nonlinearity=layer_nonlinearity,
                # decoder_nonlinearity=lasagne.nonlinearities.identity,
                decoder_nonlinearity=lasagne.nonlinearities.sigmoid,
                objective_to_minimize=objective_to_minimize,
                # objective_to_minimize=lasagne.objectives.binary_crossentropy,
                corruption_level=layer_corruption_level,
                
                W_encode=hidden_layer.W,
                b_encoder=hidden_layer.b,
                )
            
            L1_regularizer_lambda = L1_regularizer_lambdas[layer_index];
            L2_regularizer_lambda = L2_regularizer_lambdas[layer_index];
            denoising_auto_encoder.set_L1_regularizer_lambda(L1_regularizer_lambda)
            denoising_auto_encoder.set_L2_regularizer_lambda(L2_regularizer_lambda)
            
            denoising_auto_encoders.append(denoising_auto_encoder);
        
        self.network = network;
            
        self.denoising_auto_encoders = denoising_auto_encoders;
