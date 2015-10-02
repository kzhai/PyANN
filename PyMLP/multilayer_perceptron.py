"""
This tutorial introduces the multilayer perceptron using Theano.

 A multilayer perceptron is a logistic regressor where
instead of feeding the input to the logistic regression you insert a
intermediate layer, called the hidden layer, that has a nonlinear
activation function (usually tanh or sigmoid) . One can use many such
hidden layers making the architecture deep. The tutorial will also tackle
the problem of MNIST digit classification.

.. math::

    f(x) = G( b^{(2)} + W^{(2)}( s( b^{(1)} + W^{(1)} x))),

References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 5

"""

import os
import sys
import timeit

import cPickle
import numpy
import scipy

import theano
import theano.tensor
import datetime

# import layers
# import layers.dense
# import layers.noise

import lasagne
import lasagne.nonlinearities

class MultiLayerPerceptron(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softmax layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self,
                 input,
                 layer_nonlinearities=None,
                 layer_shapes=None,
                 layer_dropout_probabilities=None,
                ):
        
        if layer_dropout_probabilities is not None:
            assert (layer_dropout_probabilities[index] >= 0 for index in xrange(len(layer_dropout_probabilities)))
            assert (layer_dropout_probabilities[index] <= 1 for index in xrange(len(layer_dropout_probabilities)))
        
        layer_nonlinearities = [getattr(lasagne.nonlinearities, layer_nonlinearity) for layer_nonlinearity in layer_nonlinearities]
        
        network = lasagne.layers.InputLayer(shape=(layer_shapes[0]), input_var=input)
        
        for layer_index in xrange(1, len(layer_shapes)):
            if layer_dropout_probabilities is not None and layer_dropout_probabilities[layer_index - 1] > 0:
                network = lasagne.layers.DropoutLayer(network, p=layer_dropout_probabilities[layer_index - 1])
                
            layer_shape = layer_shapes[layer_index]
            layer_nonlinearity = layer_nonlinearities[layer_index];
            network = lasagne.layers.DenseLayer(network, layer_shape, nonlinearity=layer_nonlinearity)
        
    '''
    @staticmethod
    def load_model(input, file_path):
        file_stream = open(file_path, 'rb')
        
        
        self.input = input
        
        self._layers = [];
        
        assert len(layer_dimensionalities) >= 2;
        assert len(layer_dimensionalities) == len(activation_functions) + 1;
                 
        self._layers.append(layers.dense.DenseLayer(input, shape=(layer_dimensionalities[0], layer_dimensionalities[1]), nonlinearity=activation_functions[0]));
        
        for n_in, n_out, activation_function in zip(layer_dimensionalities[1:-1], layer_dimensionalities[2:], activation_functions[1:]):
            self._layers.append(layers.dense.DenseLayer(self._layers[-1].output, shape=(n_in, n_out), nonlinearity=activation_function));
        
        # keep track of model output
        self.output = self._layers[-1].output

        # Combine layer_parameters from all layers
        self._params = []
        for temp_layer in self._layers:
            self._params += temp_layer._params
        
        
        
        objective_to_minimize = cPickle.load(file_stream);
        #prediction_error = cPickle.load(file_stream);
        layer_activation_functions = cPickle.load(file_stream);
        layer_parameters = cPickle.load(file_stream);
        
        file_stream.close();
        
        
        def __init__(self,
                 input=None,
                 activation_functions=None,
                 layer_dimensionalities=None,
                 layer_parameters=None,
                 #objective_to_minimize=theano.tensor.nnet.categorical_crossentropy,
                 #prediction_error=neural_network_layer.prediction_error
                ):
            
        classifier = MultiLayerPerceptron(input=None,
                 activation_functions=None,
                 layer_dimensionalities=None,
                 layer_parameters=None)
        
        return classifier
    '''
