import theano
import theano.tensor
import numpy

from lasagne.layers import Layer
from lasagne import init, nonlinearities
from lasagne import nonlinearities

import network
import lasagne

from theano.tensor.shared_randomstreams import RandomStreams

def get_corruption_mask(input, corruption_level=0, rng=RandomStreams()):
    """This function keeps ``1-corruption_level`` entries of the inputs the
    same and zero-out randomly selected subset of size ``coruption_level``
    Note : first argument of theano.rng.binomial is the shape(size) of
           random numbers that it should produce
           second argument is the number of trials
           third argument is the probability of success of any trial

            this will produce an array of 0s and 1s where 1 has a
            probability of 1 - ``corruption_level`` and 0 with
            ``corruption_level``

            The binomial function return int64 data type by
            default.  int64 multiplicated by the input
            type(floatX) always return float64.  To keep all data
            in floatX when floatX is float32, we set the dtype of
            the binomial to floatX. As in our case the value of
            the binomial is always 0 or 1, this don't change the
            result. This is needed to allow the gpu to work
            correctly as it only support float32 for now.
    """
    
    return rng.binomial(size=input.shape, n=1, p=1 - corruption_level, dtype=theano.config.floatX)

class DenoisingAutoEncoderNetwork(network.Network):
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

        y = s(W_encode \tilde{x} + b)                                           (2)

        x = s(W_encode' y  + b')                                                (3)

        L(x,z) = -sum_{k=1}^d [x_k \log z_k + (1-x_k) \log( 1-z_k)]      (4)

    """
    
    def __init__(self,
            input_layer=None,
            layer_shapes=100,
            corruption_level=0,
            W_encode=init.GlorotUniform(),
            W_decode=None,
            b_encoder=init.Constant(0.),
            b_decoder=init.Constant(0.),
            encoder_nonlinearity=nonlinearities.sigmoid,
            decoder_nonlinearity=nonlinearities.sigmoid,
            **kwargs):
        
        super(DenoisingAutoEncoderNetwork, self).__init__(
        # objective_to_minimize,
        )
        
        network = input_layer;
        num_inputs = int(numpy.prod(self.input_shape[1:]))
        
        network = lasagne.layers.DropoutLayer(network, p=corruption_level);
        
        network = lasagne.layers.DenseLayer(network, layer_shapes, W = W_encode,
            b = b_encoder, nonlinearity=encoder_nonlinearity);
        
        if W_decode is None:
            W_decode = W_encode.T
        else:
            W_decode = init.GlorotUniform();
            
        network = lasagne.layers.DenseLayer(network, num_inputs, W = W_decode,
            b = b_decoder, nonlinearity=decoder_nonlinearity);
        
        self._network = network;
    
    '''
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

        activation = T.dot(input, self.W_encode)
        if self.b_encoder is not None:
            activation = activation + self.b_encoder.dimshuffle('x', 0)
        
        return self.encoder_nonlinearity(activation);

    def get_decoder_output_for(self, input):
        """
        Computes the decoder output given the encoder output
        """
        
        activation = T.dot(input, self.W_decode)
        if self.b_decoder is not None:
            activation = activation + self.b_decoder.dimshuffle('x', 0)
            
        return self.decoder_nonlinearity(activation);
    
    def get_output_for(self, input, **kwargs):
        input = self.get_corrupted_input(input);
        
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)

        return self.get_decoder_output_for(self.get_encoder_output_for(input))
    
    def get_corrupted_input(self, input):
        corruption_mask = get_corruption_mask(input, self.corruption_level);
        return corruption_mask * input
    '''