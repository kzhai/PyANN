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

class Network():
    def __init__(self):
        self.network = None;
    
    def get_output(self, inputs=None, **kwargs):
        return lasagne.layers.get_output(self.network, inputs, **kwargs)
    
    def get_output_shape(self, input_shapes=None):
        return lasagne.layers.get_output_shape(self.network, input_shapes);
    
    def get_all_layers(self, treat_as_input=None):
        return lasagne.layers.get_all_layers(self.newtork, treat_as_input=None);
    
    def get_all_params(self, **tags):
        return lasagne.layers.get_all_params(self.newtork, **tags);
    
    def count_params(self, **tags):
        return lasagne.layers.count_params(self.network, **tags);
    
    def get_all_param_values(self, **tags):
        return lasagne.layers.get_all_param_values(self.network, **tags);
    
    def set_all_param_values(self, values, **tags):
        lasagne.layers.set_all_param_values(self.network, values, **tags)