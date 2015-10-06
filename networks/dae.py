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

class DenoisingAutoEncoder(network.Network):
    def __init__(self,
            input=None,
            layer_nonlinearities=None,
            layer_shapes=None,
            layer_dropout_rates=None,
            #objective_to_minimize=network.mean_categorical_crossentropy,
            #L1_regularizer_lambdas=None,
            #L2_regularizer_lambdas=None
            ):
        super(DenoisingAutoEncoder, self).__init__(
            #objective_to_minimize,
            )
        
        network = lasagne.layers.InputLayer(shape=(None, layer_shapes[0]), input_var=input)
        #print _network.shape, _network.output_shape
        for layer_index in xrange(1, len(layer_shapes)):
            if layer_dropout_rates is not None and layer_dropout_rates[layer_index - 1] > 0:
                network = lasagne.layers.DropoutLayer(network, p=layer_dropout_rates[layer_index - 1])
                #print _network.input_shape, _network.output_shape
            
            layer_shape = layer_shapes[layer_index]
            layer_nonlinearity = layer_nonlinearities[layer_index - 1];
            network = lasagne.layers.DenseLayer(network, layer_shape, nonlinearity=layer_nonlinearity)
            #print _network.input_shape, _network.output_shape
        
        self._network = network;
        
        #self.set_L1_regularizer_lambda(L1_regularizer_lambdas);
        #self.set_L2_regularizer_lambda(L2_regularizer_lambdas);
        