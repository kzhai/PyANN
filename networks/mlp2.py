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

from networks.dae2 import DenoisingAutoEncoderNetwork
# from layers.dae import DenoisingAutoEncoderLayer

# from network import mean_categorical_crossentropy

# from layers.dae import DenoisingAutoEncoderLayer

class MultiLayerPerceptron2(network.Network):
    def __init__(self,
            input=None,
            layer_shapes=None,
            layer_nonlinearities=None,
            objective_to_minimize=None,
            L1_regularizer_lambdas=None,
            L2_regularizer_lambdas=None,
            ):
        self._input = input;

        network = lasagne.layers.InputLayer(shape=(None, layer_shapes[0]), input_var=input)
        # print _network.shape, _network.output_shape
        for layer_index in xrange(1, len(layer_shapes)):
            # network = lasagne.layers.DropoutLayer(network, p=layer_dropout_rates[layer_index - 1])
            
            layer_shape = layer_shapes[layer_index]
            layer_nonlinearity = layer_nonlinearities[layer_index - 1];
            network = lasagne.layers.DenseLayer(network, layer_shape, nonlinearity=layer_nonlinearity)
            
        self._network = network;

        assert objective_to_minimize != None;
        self._objective_to_minimize = objective_to_minimize;
        
        self.set_L1_regularizer_lambda(L1_regularizer_lambdas);
        self.set_L2_regularizer_lambda(L2_regularizer_lambdas);
        
    def __build_pretrain_network(self,
                                 layer_corruption_levels,
                                 L1_regularizer_lambdas=None,
                                 L2_regularizer_lambdas=None
                                 ):
        layers = self.get_all_layers();
        
        denoising_auto_encoders = [];
        
        assert len(layer_corruption_levels) == len(layers) - 2;
        for hidden_layer_index in xrange(1, len(layers) - 1):
            hidden_layer = layers[hidden_layer_index];
            hidden_layer_shape = hidden_layer.num_units;
            hidden_layer_nonlinearity = hidden_layer.nonlinearity
            
            layer_corruption_level = layer_corruption_levels[hidden_layer_index - 1];
            
            denoising_auto_encoder = DenoisingAutoEncoderNetwork(
                input_layer=hidden_layer.input_layer,
                layer_shape=hidden_layer_shape,
                encoder_nonlinearity=hidden_layer_nonlinearity,
                decoder_nonlinearity=lasagne.nonlinearities.sigmoid,
                # objective_to_minimize=lasagne.objectives.binary_crossentropy,
                objective_to_minimize=theano.tensor.nnet.binary_crossentropy,
                # objective_to_minimize=lasagne.objectives.binary_crossentropy,
                corruption_level=layer_corruption_level,
                L1_regularizer_lambdas=L1_regularizer_lambdas,
                L2_regularizer_lambdas=L2_regularizer_lambdas,
                W_encode=hidden_layer.W,
                b_encoder=hidden_layer.b,
                )
            
            denoising_auto_encoders.append(denoising_auto_encoder);

        return denoising_auto_encoders;

    def pretrain(self, data_x, layer_corruption_levels=None, learning_rate=1e-3, momentum=0.9, number_of_epochs=10, minibatch_size=1):
        denoising_auto_encoders = self.__build_pretrain_network(layer_corruption_levels);
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
                inputs=[self._input],
                outputs=[pretrain_loss,
                         self._input,
                         # denoising_auto_encoder._network.get_encoder_output_for(self._input),
                         # denoising_auto_encoder._network.get_decoder_output_for(self._input),
                         # denoising_auto_encoder._network.get_output_for(self._input)
                         lasagne.layers.get_output(denoising_auto_encoder._network, self._input),
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
                    
                    # print numpy.max(minibatch_x), numpy.min(minibatch_x)
                    
                    function_output = pretrain_functions[dae_index](minibatch_x)
                    temp_average_pretrain_loss = function_output[0];
                    # print temp_average_pretrain_loss
                    
                    # print function_output[1]
                    # print function_output[2]
                    # print function_output[3]
                    # print function_output[4]
                    
                    average_pretrain_loss.append(temp_average_pretrain_loss)
                
                end_time = time.time()
                
                print 'pre-training layer %i, epoch %d, average cost %f, time elapsed %f' % (dae_index + 1, pretrain_epoch_index, numpy.mean(average_pretrain_loss), end_time - start_time)
