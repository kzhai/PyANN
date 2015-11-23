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

from layers.dropout import LatentFeatureDropoutLayer

from networks.dae import DenoisingAutoEncoderNetwork

class LatentFeatureMultiLayerPerceptron(mlp.MultiLayerPerceptron):
    def __init__(self,
            input=None,
            layer_shapes=None,
            layer_nonlinearities=None,
            layer_latent_feature_alphas=None,
            objective_to_minimize=None,
            ):
        self._input = input;
        
        assert len(layer_shapes) == len(layer_nonlinearities) + 1
        assert len(layer_latent_feature_alphas) == len(layer_nonlinearities)
        
        network = lasagne.layers.InputLayer(shape=(None, layer_shapes[0]), input_var=input)
        for layer_index in xrange(1, len(layer_shapes)):
            network = LatentFeatureDropoutLayer(network, alpha=layer_latent_feature_alphas[layer_index - 1])
            
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