import theano
import theano.tensor as T
import numpy as np

from lasagne.layers import Layer
from lasagne import init
from lasagne import nonlinearities

from theano.tensor.shared_randomstreams import RandomStreams
import lasagne

import network

class BidirectionalRecurrentLayer(Layer):
    """Bidirectional Recurrent Layer (BiRNN)
    """
    
    def __init__(self,
                 incoming,
                 num_units,
                 forward_nonlinearity=nonlinearities.sigmoid,
                 backward_nonlinearity=nonlinearities.sigmoid,
                 mask_input=None,
                 **kwargs):
        '''
        W_encoder=init.GlorotUniform(gain=4.0),
        W_decoder=None,
        b_encoder=init.Constant(0.),
        b_decoder=init.Constant(0.),
        '''

        super(BidirectionalRecurrentLayer, self).__init__(incoming, **kwargs)

        forward_rnn_layer = lasagne.layers.RecurrentLayer(self.input_layer,
                                                          num_units,
                                                          W_in_to_hid=lasagne.init.GlorotUniform(
                                                              gain=network.GlorotUniformGain[forward_nonlinearity]),
                                                          W_hid_to_hid=lasagne.init.GlorotUniform(
                                                              gain=network.GlorotUniformGain[forward_nonlinearity]),
                                                          b=lasagne.init.Constant(0.),
                                                          nonlinearity=forward_nonlinearity,
                                                          hid_init=lasagne.init.Constant(0.),
                                                          backwards=False,
                                                          learn_init=False,
                                                          gradient_steps=-1,
                                                          grad_clipping=0,
                                                          unroll_scan=False,
                                                          precompute_input=True,
                                                          mask_input=mask_input,
                                                          # only_return_final=True
                                                          );

        backward_rnn_layer = lasagne.layers.RecurrentLayer(self.input_layer,
                                                           num_units,
                                                           W_in_to_hid=lasagne.init.GlorotUniform(
                                                               gain=network.GlorotUniformGain[backward_nonlinearity]),
                                                           W_hid_to_hid=lasagne.init.GlorotUniform(
                                                               gain=network.GlorotUniformGain[backward_nonlinearity]),
                                                           b=lasagne.init.Constant(0.),
                                                           nonlinearity=backward_nonlinearity,
                                                           hid_init=lasagne.init.Constant(0.),
                                                           backwards=True,
                                                           learn_init=False,
                                                           gradient_steps=-1,
                                                           grad_clipping=0,
                                                           unroll_scan=False,
                                                           precompute_input=True,
                                                           mask_input=mask_input,
                                                           # only_return_final=True
                                                           );


        self.encoder_nonlinearity = (nonlinearities.identity if encoder_nonlinearity is None
                             else encoder_nonlinearity)
        self.decoder_nonlinearity = (nonlinearities.identity if decoder_nonlinearity is None
                             else decoder_nonlinearity)

        self.corruption_level = corruption_level;

        self.num_units = num_units
        
        num_inputs = int(np.prod(self.input_shape[1:]))
        
        self.W_encoder = self.add_param(W_encoder, (num_inputs, num_units), name="W_encoder")
        
        if W_decoder is None:
            self.W_decoder = self.W_encoder.T
        else:
            self.W_decoder = self.add_param(W_decoder, (num_units, num_inputs), name="W_decoder")
        
        self.b_encoder = self.add_param(b_encoder, (num_units,), name="b_encoder", regularizable=False)
        
        self.b_decoder = self.add_param(b_decoder, (num_inputs,), name="b_decoder", regularizable=False)

    def get_decoder_shape_for(self, input_shape):
        return input_shape
    
    def get_encoder_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)
    
    def get_output_shape_for(self, input_shape):
        return self.get_decoder_shape_for(input_shape);

    def get_encoder_output_for(self, input):
        """
        Computes the encoder output given the input
        """
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)

        activation = T.dot(input, self.W_encoder)
        if self.b_encoder is not None:
            activation = activation + self.b_encoder.dimshuffle('x', 0)
        
        return self.encoder_nonlinearity(activation);

    def get_decoder_output_for(self, input):
        """
        Computes the decoder output given the encoder output
        """
        
        activation = T.dot(input, self.W_decoder)
        if self.b_decoder is not None:
            activation = activation + self.b_decoder.dimshuffle('x', 0)
            
        return self.decoder_nonlinearity(activation);
    
    def get_output_for(self, input, **kwargs):
        '''
        if 'corruption_level' in kwargs:
            corruption_level = kwargs['corruption_level'];
            print "corruption_level:", corruption_level 
            filter_mask = get_filter_mask(input, 1 - corruption_level);
            input *= filter_mask
        '''
        
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)
        
        filter_mask = get_filter_mask(input, 1 - self.corruption_level);
        input_tilde = input * filter_mask
        
        return self.get_decoder_output_for(self.get_encoder_output_for(input_tilde))
