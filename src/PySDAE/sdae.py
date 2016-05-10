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

# from layers.dae import DenoisingAutoEncoderLayer
from PyDAE.dae import DenoisingAutoEncoder

from theano.tensor.shared_randomstreams import RandomStreams

class StackedDenoisingAutoEncoder(network.Network):
    def __init__(self,
            input_network=None,
            layer_shapes=None,
            layer_nonlinearities=None,
            layer_corruption_levels=None,
            objective_to_minimize=lasagne.objectives.binary_crossentropy,
            L1_regularizer_lambdas=None,
            L2_regularizer_lambdas=None,
            ):
        super(StackedDenoisingAutoEncoder, self).__init__(input_network)

        assert len(layer_shapes) == len(layer_nonlinearities)
        assert len(layer_shapes) == len(layer_corruption_levels)
        # assert len(layer_activation_parameters) == len(layer_nonlinearities)
        # assert len(layer_activation_styles) == len(layer_nonlinearities)
        assert len(layer_shapes) == len(L1_regularizer_lambdas)
        assert len(layer_shapes) == len(L2_regularizer_lambdas)
        
        neural_network = input_network;
        denoising_auto_encoders = [];
        for layer_index in xrange(len(layer_shapes)):
            layer_shape = layer_shapes[layer_index]
            layer_nonlinearity = layer_nonlinearities[layer_index];
            
            input_layer = neural_network;
            neural_network = lasagne.layers.DenseLayer(neural_network, layer_shape, W=lasagne.init.GlorotUniform(gain=network.GlorotUniformGain[layer_nonlinearity]), nonlinearity=layer_nonlinearity)
            
            #print neural_network.W.get_value()
            #print neural_network.b.get_value()
            
            #print type(neural_network.W)
            #print type(neural_network.b)
            
            layer_corruption_level = layer_corruption_levels[layer_index];
            denoising_auto_encoder = DenoisingAutoEncoder(
                input_layer,
                layer_shape,
                
                W_encoder=neural_network.W,
                b_encoder=neural_network.b,
                
                encoder_nonlinearity=layer_nonlinearity,
                # decoder_nonlinearity=layer_nonlinearity,
                # decoder_nonlinearity=lasagne.nonlinearities.identity,
                decoder_nonlinearity=lasagne.nonlinearities.sigmoid,
                objective_to_minimize=objective_to_minimize,
                corruption_level=layer_corruption_level,
                )
            
            L1_regularizer_lambda = L1_regularizer_lambdas[layer_index];
            L2_regularizer_lambda = L2_regularizer_lambdas[layer_index];
            denoising_auto_encoder.set_L1_regularizer_lambda(L1_regularizer_lambda)
            denoising_auto_encoder.set_L2_regularizer_lambda(L2_regularizer_lambda)
            
            denoising_auto_encoders.append(denoising_auto_encoder);
        
        self.network = neural_network;
            
        self.denoising_auto_encoders = denoising_auto_encoders;
