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

from layers.dropout import AdaptiveDropoutLayer

# from networks.dae import DenoisingAutoEncoder

class MultiLayerPerceptron(network.Network):
    def __init__(self,
            input_network=None,
            layer_dimensions=None,
            layer_nonlinearities=None,
            input_activation_rate=1.0,
            #layer_activation_parameters=None,
            #layer_activation_styles=None,
            objective_to_minimize=None,
            # pretrained_model=None,
            ):
        super(MultiLayerPerceptron, self).__init__(input_network)

        assert len(layer_dimensions) == len(layer_nonlinearities)
        #assert len(layer_dimensions) == len(layer_activation_parameters)
        #assert len(layer_dimensions) == len(layer_activation_styles)
        
        '''
        pretrained_network_layers = None;
        if pretrained_model != None:
            pretrained_network_layers = lasagne.layers.get_all_layers(pretrained_model._neural_network);
        '''

        #dropout_network = input_network
        neural_network = input_network;
        for layer_index in xrange(len(layer_dimensions)):
            previous_layer_dimension = lasagne.layers.get_output_shape(neural_network)[1:];
            #activation_probability = sample_activation_probability(previous_layer_dimension, layer_activation_styles[layer_index], layer_activation_parameters[layer_index]);

            if layer_index==0:
                #activation_probability = sample_activation_probability(previous_layer_dimension, layer_activation_styles[0], layer_activation_parameters[0]);
                #neural_network = GeneralizedDropoutLayer(neural_network, activation_probability=activation_probability);
                neural_network = lasagne.layers.DropoutLayer(neural_network, p=1-input_activation_rate);

            layer_dimension = layer_dimensions[layer_index]
            layer_nonlinearity = layer_nonlinearities[layer_index];
            
            dense_layer = lasagne.layers.DenseLayer(neural_network,
                                                    layer_dimension,
                                                    W=lasagne.init.GlorotUniform(gain=network.GlorotUniformGain[layer_nonlinearity]),
                                                    nonlinearity=layer_nonlinearity)

            if layer_index<len(layer_dimensions)-1:
                dropout_layer = AdaptiveDropoutLayer(neural_network,
                                                     layer_dimension,
                                                     W=dense_layer.W)
                neural_network = lasagne.layers.ElemwiseMergeLayer([dense_layer, dropout_layer], theano.tensor.mul);
            else:
                neural_network = dense_layer;
            
            '''
            if pretrained_network_layers == None or len(pretrained_network_layers) <= layer_index:
                _neural_network = lasagne.layers.DenseLayer(_neural_network, layer_dimension, nonlinearity=layer_nonlinearity)
            else:
                pretrained_layer = pretrained_network_layers[layer_index];
                assert isinstance(pretrained_layer, lasagne.layers.DenseLayer)
                assert pretrained_layer.nonlinearity == layer_nonlinearity, (pretrained_layer.nonlinearity, layer_nonlinearity)
                assert pretrained_layer.num_units == layer_dimension
                
                _neural_network = lasagne.layers.DenseLayer(_neural_network,
                                                    layer_dimension,
                                                    W=pretrained_layer.W,
                                                    b=pretrained_layer.b,
                                                    nonlinearity=layer_nonlinearity)
            '''

        self._neural_network = neural_network;

        assert objective_to_minimize != None;
        self._objective_to_minimize = objective_to_minimize;

    def get_objective_to_minimize(self, label, **kwargs):
        output = self.get_output(**kwargs);

        if "objective_to_minimize" in kwargs:
            objective_to_minimize = getattr(lasagne.objectives, kwargs["objective_to_minimize"]);
            minimization_objective = theano.tensor.mean(objective_to_minimize(output, label), dtype=theano.config.floatX);
        else:
            minimization_objective = theano.tensor.mean(self._objective_to_minimize(output, label), dtype=theano.config.floatX)

        minimization_objective += self.L1_regularizer()
        minimization_objective += self.L2_regularizer();

        minimization_objective += self.dae_regularizer();
        
        return minimization_objective
    
    def dae_regularizer(self):
        if self._layer_dae_regularizer_lambdas == None:
            return 0;
        else:
            dae_regularization = 0;
            for dae_layer in self._layer_dae_regularizer_lambdas:
                dae_regularization += self._layer_dae_regularizer_lambdas[dae_layer] * dae_layer.get_objective_to_minimize()
            return dae_regularization;

    def set_dae_regularizer_lambda(self,
                                   layer_dae_regularizer_lambdas,
                                   layer_corruption_levels=None,
                                   L1_regularizer_lambdas=None,
                                   L2_regularizer_lambdas=None
                                   ):
        if layer_dae_regularizer_lambdas == None or all(layer_dae_regularizer_lambda == 0 for layer_dae_regularizer_lambda in layer_dae_regularizer_lambdas):
            self._layer_dae_regularizer_lambdas = None;
        else:
            assert len(layer_dae_regularizer_lambdas) == (len(self.get_all_layers()) - 1) / 2 - 1;
            dae_regularizer_layers = self.__build_dae_network(layer_corruption_levels, L1_regularizer_lambdas, L2_regularizer_lambdas);
            self._layer_dae_regularizer_lambdas = {temp_layer:layer_dae_regularizer_lambda for temp_layer, layer_dae_regularizer_lambda in zip(dae_regularizer_layers, layer_dae_regularizer_lambdas)};
            
        return;