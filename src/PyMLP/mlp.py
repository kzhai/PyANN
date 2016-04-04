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

from layers.dropout import GeneralizedDropoutLayer, sample_activation_probability

# from networks.dae import DenoisingAutoEncoder

class MultiLayerPerceptron(network.Network):
    def __init__(self,
            network=None,
            layer_shapes=None,
            layer_nonlinearities=None,
            layer_activation_parameters=None,
            layer_activation_styles=None,
            objective_to_minimize=None,
            pretrained_model=None
            ):
        self.input = lasagne.layers.get_output(network);
        
        assert len(layer_shapes) == len(layer_nonlinearities)
        assert len(layer_shapes) == len(layer_activation_parameters)
        assert len(layer_shapes) == len(layer_activation_styles) 
        
        pretrained_network_layers = None;
        if pretrained_model != None:
            pretrained_network_layers = lasagne.layers.get_all_layers(pretrained_model.network);
        
        for layer_index in xrange(1, len(layer_shapes)):
            previous_layer_shape = layer_shapes[layer_index - 1]
            
            activation_probability = sample_activation_probability(previous_layer_shape, layer_activation_styles[layer_index - 1], layer_activation_parameters[layer_index - 1]);
            
            network = GeneralizedDropoutLayer(network, activation_probability=activation_probability);
            
            layer_shape = layer_shapes[layer_index]
            layer_nonlinearity = layer_nonlinearities[layer_index - 1];
            
            if pretrained_network_layers == None or len(pretrained_network_layers) <= layer_index:
                network = lasagne.layers.DenseLayer(network, layer_shape, nonlinearity=layer_nonlinearity)
            else:
                pretrained_layer = pretrained_network_layers[layer_index];
                assert isinstance(pretrained_layer, lasagne.layers.DenseLayer)
                assert pretrained_layer.nonlinearity == layer_nonlinearity, (pretrained_layer.nonlinearity, layer_nonlinearity)
                assert pretrained_layer.num_units == layer_shape
                
                network = lasagne.layers.DenseLayer(network,
                                                    layer_shape,
                                                    W=pretrained_layer.W,
                                                    b=pretrained_layer.b,
                                                    nonlinearity=layer_nonlinearity) 
                
        self.network = network;

        assert objective_to_minimize != None;
        self.objective_to_minimize = objective_to_minimize;

    def get_objective_to_minimize(self, label):
        train_loss = theano.tensor.mean(self.objective_to_minimize(self.get_output(), label))
        train_loss += self.L1_regularizer();
        train_loss += self.L2_regularizer();
        
        train_loss += self.dae_regularizer();
        
        return train_loss
    
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
    
    def __build_dae_network(self,
                            layer_corruption_levels=None,
                            L1_regularizer_lambdas=None,
                            L2_regularizer_lambdas=None
                            ):
        layers = self.get_all_layers();
        
        denoising_auto_encoders = [];
        
        if layer_corruption_levels is None:
            layer_corruption_levels = numpy.zeros((len(layers) - 1) / 2 - 1)
        assert len(layer_corruption_levels) == (len(layers) - 1) / 2 - 1;
            
        for hidden_layer_index in xrange(2, len(layers) - 1, 2):
            hidden_layer = layers[hidden_layer_index];
            # this is to get around the dropout layer
            # input = hidden_layer.input
            input_layer = layers[hidden_layer_index - 2];
            hidden_layer_shape = hidden_layer.num_units;
            hidden_layer_nonlinearity = hidden_layer.nonlinearity
            
            # this is to get around the dropout layer
            # layer_corruption_level = layer_corruption_levels[hidden_layer_index - 1];
            layer_corruption_level = layer_corruption_levels[hidden_layer_index / 2 - 1];
            
            denoising_auto_encoder = DenoisingAutoEncoder(
                input=input_layer,
                layer_shape=hidden_layer_shape,
                encoder_nonlinearity=hidden_layer_nonlinearity,
                decoder_nonlinearity=lasagne.nonlinearities.sigmoid,
                # objective_to_minimize=lasagne.objectives.binary_crossentropy,
                objective_to_minimize=theano.tensor.nnet.binary_crossentropy,
                # objective_to_minimize=lasagne.objectives.binary_crossentropy,
                corruption_level=layer_corruption_level,
                # L1_regularizer_lambdas=L1_regularizer_lambdas,
                # L2_regularizer_lambdas=L2_regularizer_lambdas,
                W_encode=hidden_layer.W,
                b_encoder=hidden_layer.b,
                )
            
            denoising_auto_encoders.append(denoising_auto_encoder);

        return denoising_auto_encoders;
    
    """
    def pretrain_with_dae(self, data_x, layer_corruption_levels=None, number_of_epochs=50, minibatch_size=1, learning_rate=1e-3, momentum=0.95):
        denoising_auto_encoders = self.__build_dae_network(layer_corruption_levels);
        pretrain_functions = [];
        for denoising_auto_encoder in denoising_auto_encoders:
            '''
            train_prediction = lasagne.layers.get_output(denoising_auto_encoder)
            pretrain_loss = x * theano.tensor.log(train_prediction) + (1 - x) * theano.tensor.log(1 - train_prediction)
            pretrain_loss = theano.tensor.mean(-theano.tensor.sum(pretrain_loss, axis=1))
            '''
            
            pretrain_loss = denoising_auto_encoder.get_objective_to_minimize();
            # pretrain_loss = theano.tensor.mean(theano.tensor.nnet.categorical_crossentropy(train_prediction, y))
            
            '''
            # We could add some weight decay as well here, see lasagne.regularization.
            dae_layers = lasagne.layers.get_all_layers(denoising_auto_encoders);
            L1_regularizer_layer_lambdas = {temp_layer:L1_regularizer_lambda for temp_layer, L1_regularizer_lambda in zip(dae_layers[1:], L1_regularizer_lambdas)};
            L1_regularizer = lasagne.regularization.regularize_layer_params_weighted(L1_regularizer_layer_lambdas, lasagne.regularization.l1)
            L2_regularizer_layer_lambdas = {temp_layer:L2_regularizer_lambda for temp_layer, L2_regularizer_lambda in zip(dae_layers[1:], L2_regularizer_lambdas)};
            L2_regularizer = lasagne.regularization.regularize_layer_params_weighted(L2_regularizer_layer_lambdas, lasagne.regularization.l2)
            pretrain_loss += L1_regularizer + L2_regularizer
            '''
        
            # Create update expressions for training, i.e., how to modify the
            # parameters at each training step. Here, we'll use Stochastic Gradient
            # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
            all_dae_params = denoising_auto_encoder.get_all_params(trainable=True)
            # all_dae_params = lasagne.layers.get_all_params(denoising_auto_encoder, trainable=True)
            updates = lasagne.updates.nesterov_momentum(pretrain_loss, all_dae_params, learning_rate, momentum)
        
            '''
            # Create a pretrain_loss expression for validation/testing. The crucial difference
            # here is that we do a deterministic forward pass through the networks,
            # disabling dropout layers.
            validate_prediction = lasagne.layers.get_output(networks, deterministic=True)
            validate_loss = theano.tensor.mean(theano.tensor.nnet.categorical_crossentropy(validate_prediction, y))
            # As a bonus, also create an expression for the classification accuracy:
            validate_accuracy = theano.tensor.mean(theano.tensor.eq(theano.tensor.argmax(validate_prediction, axis=1), y), dtype=theano.config.floatX)
            '''
            
            # Compile a function performing a training step on a mini-batch (by giving
            # the updates dictionary) and returning the corresponding training pretrain_loss:
            pretrain_function = theano.function(
                inputs=[self.input],
                outputs=[pretrain_loss,
                         self.input,
                         # denoising_auto_encoder.network.get_encoder_output_for(self.input),
                         # denoising_auto_encoder.network.get_decoder_output_for(self.input),
                         # denoising_auto_encoder.network.get_output_for(self.input)
                         lasagne.layers.get_output(denoising_auto_encoder.network, self.input),
                         ],
                updates=updates
            )
            
            pretrain_functions.append(pretrain_function);
        
        number_of_minibatches_to_pretrain = data_x.shape[0] / minibatch_size
    
        # start_time = timeit.default_timer()
        for dae_index in xrange(len(denoising_auto_encoders)):
            # denoising_auto_encoder = denoising_auto_encoders[dae_index]
            # layer_corruption_level = layer_corruption_levels[dae_index]
            for pretrain_epoch_index in xrange(number_of_epochs):
                start_time = time.time()
                 
                average_pretrain_loss = []
                for minibatch_index in xrange(number_of_minibatches_to_pretrain):
                    iteration_index = pretrain_epoch_index * number_of_minibatches_to_pretrain + minibatch_index
                
                    minibatch_x = data_x[minibatch_index * minibatch_size:(minibatch_index + 1) * minibatch_size, :]
                    
                    function_output = pretrain_functions[dae_index](minibatch_x)
                    temp_average_pretrain_loss = function_output[0];
                    # print temp_average_pretrain_loss
                    
                    average_pretrain_loss.append(temp_average_pretrain_loss)
                
                end_time = time.time()
                
                print 'pre-training layer %i, epoch %d, average cost %f, time elapsed %f' % (dae_index + 1, pretrain_epoch_index, numpy.mean(average_pretrain_loss), end_time - start_time)
    """
