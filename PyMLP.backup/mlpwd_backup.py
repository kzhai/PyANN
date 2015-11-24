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
import mlp

from layers.dropout import GeneralizedDropoutLayer

from networks.dae import DenoisingAutoEncoderNetwork

class GeneralizedMultiLayerPerceptron(mlp.MultiLayerPerceptron):
    def __init__(self,
            input=None,
            layer_shapes=None,
            layer_nonlinearities=None,
            layer_dropout_parameters=None,
            layer_dropout_styles=None,
            objective_to_minimize=None,
            latent_feature_model="beta"
            ):
        self._input = input;
        
        assert len(layer_shapes) == len(layer_nonlinearities) + 1
        assert len(layer_dropout_parameters) == len(layer_nonlinearities)
        assert len(layer_dropout_styles) == len(layer_nonlinearities)
        
        network = lasagne.layers.InputLayer(shape=(None, layer_shapes[0]), input_var=input)
        for layer_index in xrange(1, len(layer_shapes)):
            if latent_feature_model == "bernoulli":
                previous_layer_shape = layer_shapes[layer_index - 1]
                activation_probability = numpy.zeros(previous_layer_shape) + layer_dropout_parameters[layer_index - 1];
                network = GeneralizedDropoutLayer(network, activation_probability=activation_probability);
            elif latent_feature_model == "beta-bernoulli":
                previous_layer_shape = layer_shapes[layer_index - 1]
                
                shape_alpha = layer_dropout_parameters[layer_index - 1] / numpy.arange(1, previous_layer_shape + 1);
                shape_beta = 1.0;
                
                activation_probability = numpy.zeros(previous_layer_shape);
                for index in xrange(previous_layer_shape):
                    activation_probability[index] = numpy.random.beta(shape_alpha[index], shape_beta);
                
                network = GeneralizedDropoutLayer(network, activation_probability=activation_probability);
            elif latent_feature_model == "reciprocal":
                previous_layer_shape = layer_shapes[layer_index - 1]
                activation_probability = layer_dropout_parameters[layer_index - 1] / numpy.arange(1, previous_layer_shape + 1);
                
                network = GeneralizedDropoutLayer(network, activation_probability=activation_probability);
            else:
                sys.stderr.write("erro: unrecognized configuration...\n");
                sys.exit();
            
            layer_shape = layer_shapes[layer_index]
            layer_nonlinearity = layer_nonlinearities[layer_index - 1];
            network = lasagne.layers.DenseLayer(network, layer_shape, nonlinearity=layer_nonlinearity)
            
        self._network = network;

        assert objective_to_minimize != None;
        self._objective_to_minimize = objective_to_minimize;
        
        '''
        self.set_L1_regularizer_lambda(L1_regularizer_lambdas);
        self.set_L2_regularizer_lambda(L2_regularizer_lambdas);
        '''
