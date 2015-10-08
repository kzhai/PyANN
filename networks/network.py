import abc
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

def mean_categorical_crossentropy(network, label):
    # Create a train_loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy train_loss):
    train_prediction = network.get_output();
    train_loss = theano.tensor.mean(theano.tensor.nnet.categorical_crossentropy(train_prediction, label))
    
    return train_loss

def mean_binary_crossentropy(network, label):
    # Create a train_loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy train_loss):
    train_prediction = network.get_output();
    train_loss = theano.tensor.mean(theano.tensor.nnet.binary_crossentropy(train_prediction, label))
    
    return train_loss
    
def mean_accuracy(network, label):
    train_prediction = network.get_output();
    train_accuracy = theano.tensor.mean(theano.tensor.eq(theano.tensor.argmax(train_prediction, axis=1), label), dtype=theano.config.floatX)
    
    return train_accuracy

'''
def updates(loss_function, learning_rate):
    all_network_params = network.get_all_params(trainable=True)
    updates = lasagne.updates.nesterov_momentum(loss_function, all_network_params, learning_rate, momentum=0.9)
    
    return updates
'''

class Network(object):
    __metaclass__ = abc.ABCMeta
    
    def __init__(self,
            objective_to_minimize=theano.tensor.nnet.categorical_crossentropy,
            #objective_to_minimize=mean_categorical_crossentropy,
            #updates_to_parameters=lasagne.updates.nesterov_momentum,
            #L1_regularizer_lambdas=None,
            #L2_regularizer_lambdas=None
            ):
        self._network = None;
        
        self._objective_to_minimize = objective_to_minimize;
        #self._updates_to_parameters = updates_to_parameters;
        
        #self.set_L1_regularizer_lambda(L1_regularizer_lambdas);
        #self.set_L2_regularizer_lambda(L2_regularizer_lambdas);

    def get_objective_to_minimize(self, label):
        train_loss = theano.tensor.mean(self._objective_to_minimize(self.get_output(), label))
        train_loss += self.L1_regularizer()
        train_loss += self.L2_regularizer();
        
        return train_loss
    
    def L1_regularizer(self):
        if self._layer_L1_regularizer_lambdas == None:
            return 0;
        else:
            # We could add some weight decay as well here, see lasagne.regularization.
            return lasagne.regularization.regularize_layer_params_weighted(self._layer_L1_regularizer_lambdas, lasagne.regularization.l1)

    def set_L1_regularizer_lambda(self, L1_regularizer_lambdas=None):
        if L1_regularizer_lambdas == None:
            self._layer_L1_regularizer_lambdas = None;
        else:
            assert len(L1_regularizer_lambdas) == len(self.get_all_layers()) - 1;
            self._layer_L1_regularizer_lambdas = {temp_layer:L1_regularizer_lambda for temp_layer, L1_regularizer_lambda in zip(self._network.get_all_layers(), self._L1_regularizer_lambdas)};
        
    def L2_regularizer(self):
        if self._layer_L2_regularizer_lambdas == None:
            return 0;
        else:
            # We could add some weight decay as well here, see lasagne.regularization.
            return lasagne.regularization.regularize_layer_params_weighted(self._layer_L2_regularizer_lambdas, lasagne.regularization.l2)

    def set_L2_regularizer_lambda(self, L2_regularizer_lambdas):
        if L2_regularizer_lambdas == None:
            self._layer_L2_regularizer_lambdas = None;
        else:
            assert len(L2_regularizer_lambdas) == len(self.get_all_layers()) - 1;
            
            self._layer_L2_regularizer_lambdas = {temp_layer:L2_regularizer_lambda for temp_layer, L2_regularizer_lambda in zip(self._network.get_all_layers(), self._L2_regularizer_lambdas)};
    
    def get_output(self, inputs=None, **kwargs):
        return lasagne.layers.get_output(self._network, inputs, **kwargs)
    
    def get_output_shape(self, input_shapes=None):
        return lasagne.layers.get_output_shape(self._network, input_shapes);
    
    def get_all_layers(self, treat_as_input=None):
        return lasagne.layers.get_all_layers(self._network, treat_as_input=None);
    
    def get_all_params(self, **tags):
        return lasagne.layers.get_all_params(self._network, **tags);
    
    def count_params(self, **tags):
        return lasagne.layers.count_params(self._network, **tags);
    
    def get_all_param_values(self, **tags):
        return lasagne.layers.get_all_param_values(self._network, **tags);
    
    def set_all_param_values(self, values, **tags):
        lasagne.layers.set_all_param_values(self._network, values, **tags)
    
    def set_input_variable(self, input):
        '''This is to establish the computational graph'''
        self.get_all_layers()[0].input_var = input
