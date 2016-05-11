import theano
import theano.tensor
import numpy

from lasagne.layers import Layer
from lasagne import init, nonlinearities
from lasagne import nonlinearities

import lasagne

import network

from layers.dae import DenoisingAutoEncoderLayer

from theano.tensor.shared_randomstreams import RandomStreams

class DenoisingAutoEncoder(network.Network):
    """Denoising Auto-Encoder class (dA)

    A denoising autoencoders tries to reconstruct the input from a corrupted
    version of it by projecting it first in a latent space and reprojecting
    it afterwards back in the input space. Please refer to Vincent et al.,2008
    for more details. If x is the input then equation (1) computes a partially
    destroyed version of x by means of a stochastic mapping q_D. Equation (2)
    computes the projection of the input into the latent space. Equation (3)
    computes the reconstruction of the input, while equation (4) computes the
    reconstruction error.

    .. math::

        \tilde{x} ~ q_D(\tilde{x}|x)                                     (1)

        y = s(W_encode \tilde{x} + b)                                    (2)

        x = s(W_encode' y  + b')                                         (3)

        L(x,z) = -sum_{k=1}^d [x_k \log z_k + (1-x_k) \log( 1-z_k)]      (4)
    """
    def __init__(self,
            input_network,
            layer_dimension,

            # In theano denoising auto-encoder example, it is initialized to init.GlorotUniform(gain=4.0).
            W_encoder=init.GlorotUniform(gain=4.0),
            W_decoder=None,
            b_encoder=init.Constant(0.),
            b_decoder=init.Constant(0.),
            
            encoder_nonlinearity=lasagne.nonlinearities.sigmoid,
            decoder_nonlinearity=lasagne.nonlinearities.sigmoid,
            objective_to_minimize=lasagne.objectives.squared_error,
            corruption_level=0,
            **kwargs):
        super(DenoisingAutoEncoder, self).__init__(input_network)
        
        # print "----------"
        # print type(W_encoder)
        # print type(W_decoder)
        # print type(b_encoder)
        # print type(b_decoder)
        
        neural_network = input_network;
        neural_network = DenoisingAutoEncoderLayer(
            neural_network,
            layer_dimension,
            corruption_level,
            W_encoder=W_encoder,
            W_decoder=W_decoder,
            b_encoder=b_encoder,
            b_decoder=b_decoder,
            encoder_nonlinearity=encoder_nonlinearity,
            decoder_nonlinearity=decoder_nonlinearity
            );
        
        # print neural_network.W_encoder.get_value();
        # print neural_network.W_decoder.get_value();
        # print neural_network.b_encoder.get_value();
        # print neural_network.b_decoder.get_value();
        # print "----------"
        
        self._neural_network = neural_network;
        
        assert objective_to_minimize != None;
        self._objective_to_minimize = objective_to_minimize;
        
    def get_objective_to_minimize(self, **kwargs):
        train_loss = theano.tensor.mean(theano.tensor.sum(self._objective_to_minimize(self.get_output(), self.get_input()), axis=1))
        
        train_loss += self.L1_regularizer()
        train_loss += self.L2_regularizer();
        
        return train_loss
    
    def get_all_params(self, **tags):
        # return lasagne.layers.get_all_params(self._neural_network, **tags);
        return self._neural_network.get_params(**tags);
