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

class MultiLayerPerceptron(network.Network):
    def __init__(self,
            input=None,
            layer_nonlinearities=None,
            layer_shapes=None,
            layer_dropout_rates=None
            ):
        # print layer_nonlinearities
        #layer_nonlinearities = [getattr(lasagne.nonlinearities, layer_nonlinearity) for layer_nonlinearity in layer_nonlinearities]
        
        network = lasagne.layers.InputLayer(shape=(None, layer_shapes[0]), input_var=input)
        #print network.shape, network.output_shape
        for layer_index in xrange(1, len(layer_shapes)):
            if layer_dropout_rates is not None and layer_dropout_rates[layer_index - 1] > 0:
                network = lasagne.layers.DropoutLayer(network, p=layer_dropout_rates[layer_index - 1])
                #print network.input_shape, network.output_shape
            
            layer_shape = layer_shapes[layer_index]
            layer_nonlinearity = layer_nonlinearities[layer_index - 1];
            network = lasagne.layers.DenseLayer(network, layer_shape, nonlinearity=layer_nonlinearity)
            #print network.input_shape, network.output_shape
        
        self.newtork = network;