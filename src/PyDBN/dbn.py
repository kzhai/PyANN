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

from PyRBM.rbm import RestrictedBoltzmannMachine

from theano.tensor.shared_randomstreams import RandomStreams

class DeepBeliefNetwork(network.Network):
    def __init__(self,
            input_network=None,
            layer_dimensions=None,
            #layer_nonlinearities=None,
            #layer_corruption_levels=None,
            #objective_to_minimize=lasagne.objectives.binary_crossentropy,
            #L1_regularizer_lambdas=None,
            #L2_regularizer_lambdas=None,
            ):
        super(DeepBeliefNetwork, self).__init__(input_network)

        # assert len(layer_dimensions) == len(layer_nonlinearities)
        # assert len(layer_dimensions) == len(layer_corruption_levels)
        # assert len(layer_activation_parameters) == len(layer_nonlinearities)
        # assert len(layer_activation_styles) == len(layer_nonlinearities)
        # assert len(layer_dimensions) == len(L1_regularizer_lambdas)
        # assert len(layer_dimensions) == len(L2_regularizer_lambdas)
        layer_nonlinearity = lasagne.nonlinearities.sigmoid;
                
        neural_network = input_network;
        restricted_boltzmann_machines = [];
        for layer_index in xrange(len(layer_dimensions)):
            layer_dimension = layer_dimensions[layer_index]
            #layer_nonlinearity = layer_nonlinearities[layer_index];
            
            input_layer = neural_network;
            neural_network = lasagne.layers.DenseLayer(neural_network, layer_dimension, W=lasagne.init.GlorotUniform(gain=network.GlorotUniformGain[layer_nonlinearity]), nonlinearity=layer_nonlinearity)

            restricted_boltzmann_machine = RestrictedBoltzmannMachine(
                input_network=input_layer,
                layer_dimension=layer_dimension,
                W=neural_network.W,
                b_hidden=neural_network.b,
                )
            
            restricted_boltzmann_machines.append(restricted_boltzmann_machine);
        
        self.network = neural_network;
        
        self.restricted_boltzmann_machines = restricted_boltzmann_machines;
