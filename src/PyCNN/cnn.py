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

class ConvolutionalNeuralNetwork(network.Network):
    def __init__(self,
            input_network=None,
            # input_shape=None,
            
            convolution_filters=None,
            convolution_nonlinearities=None,
            # convolution_filter_sizes=None,
            # maxpooling_sizes=None,
            
            dense_dimensions=None,
            dense_nonlinearities=None,
            
            dense_activation_parameters=None,
            dense_activation_styles=None,
            
            convolution_filter_size=(5, 5),
            maxpooling_size=(2, 2),
            pooling_stride=None,
            # pooling_stride=(1, 1),
            
            objective_to_minimize=None,
            ):
        super(ConvolutionalNeuralNetwork, self).__init__(input_network)

        assert len(dense_activation_parameters) == len(dense_nonlinearities) + len(convolution_nonlinearities)
        assert len(dense_activation_styles) == len(dense_nonlinearities) + len(convolution_nonlinearities)
        
        dropout_layer_index = 0;
        
        # This time we do not apply input dropout, as it tends to work less well for convolutional layers.
        assert len(convolution_filters) == len(convolution_nonlinearities);
        
        neural_network = input_network;
        for conv_layer_index in xrange(len(convolution_filters)):
            input_layer_shape = lasagne.layers.get_output_shape(neural_network)[1:]
            previous_layer_shape = numpy.prod(input_layer_shape)
            
            activation_probability = sample_activation_probability(previous_layer_shape, dense_activation_styles[dropout_layer_index], dense_activation_parameters[dropout_layer_index]);
            dropout_layer_index += 1;
            
            activation_probability = numpy.reshape(activation_probability, input_layer_shape)
            # print "before dropout", lasagne.layers.get_output_shape(neural_network)
            
            neural_network = GeneralizedDropoutLayer(neural_network, activation_probability=activation_probability);
            
            conv_filter_number = convolution_filters[conv_layer_index];
            conv_nonlinearity = convolution_nonlinearities[conv_layer_index];
            
            # conv_filter_size = convolution_filter_sizes[conv_layer_index]
            conv_filter_size = convolution_filter_size;
            
            # print "before convolution", lasagne.layers.get_output_shape(neural_network)
            # Convolutional layer with 32 kernels of size 5x5. Strided and padded convolutions are supported as well; see the docstring.
            neural_network = lasagne.layers.Conv2DLayer(neural_network,
                                                 # W=W,
                                                 num_filters=conv_filter_number,
                                                 filter_size=conv_filter_size,
                                                 nonlinearity=conv_nonlinearity,
                                                 )
            
            # pooling_size = maxpooling_sizes[conv_layer_index];
            pooling_size = maxpooling_size
            
            # print "before maxpooling", lasagne.layers.get_output_shape(neural_network)
            # Max-pooling layer of factor 2 in both dimensions:
            filter_size_for_pooling = lasagne.layers.get_output_shape(neural_network)[2:]
            if numpy.any(filter_size_for_pooling < pooling_size):
                print "warning: filter size %s is smaller than pooling size %s, skip pooling layer" % (lasagne.layers.get_output_shape(neural_network), pooling_size)
                continue;
            neural_network = lasagne.layers.MaxPool2DLayer(neural_network,
                                                    pool_size=pooling_size,
                                                    stride=pooling_stride,
                                                    )
            
        assert len(dense_dimensions) == len(dense_nonlinearities)
        for layer_index in xrange(len(dense_dimensions)):
            input_layer_shape = lasagne.layers.get_output_shape(neural_network)[1:]
            previous_layer_shape = numpy.prod(input_layer_shape)
            activation_probability = sample_activation_probability(previous_layer_shape, dense_activation_styles[dropout_layer_index], dense_activation_parameters[dropout_layer_index]);
            dropout_layer_index += 1;
            
            activation_probability = numpy.reshape(activation_probability, input_layer_shape)

            # print "before dropout", lasagne.layers.get_output_shape(neural_network)
            neural_network = GeneralizedDropoutLayer(neural_network, activation_probability=activation_probability);
            
            layer_shape = dense_dimensions[layer_index]
            layer_nonlinearity = dense_nonlinearities[layer_index];
            
            # print "before dense", lasagne.layers.get_output_shape(neural_network)
            neural_network = lasagne.layers.DenseLayer(neural_network, layer_shape, W=lasagne.init.GlorotUniform(gain=network.GlorotUniformGain[layer_nonlinearity]), nonlinearity=layer_nonlinearity)
            
        self.network = neural_network;

        assert objective_to_minimize != None;
        self.objective_to_minimize = objective_to_minimize;
    
    """
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
