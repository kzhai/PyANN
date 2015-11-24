import theano
import theano.tensor as T
import numpy as np

from lasagne.layers import Layer
from lasagne import init
from lasagne import nonlinearities

from theano.tensor.shared_randomstreams import RandomStreams

def get_corruption_mask(input, retain_probability=0, rng=RandomStreams()):
    """This function keeps ``1-retain_probability`` entries of the inputs the
    same and zero-out randomly selected subset of size ``coruption_level``
    Note : first argument of theano.rng.binomial is the shape(size) of
           random numbers that it should produce
           second argument is the number of trials
           third argument is the probability of success of any trial

            this will produce an array of 0s and 1s where 1 has a
            probability of 1 - ``retain_probability`` and 0 with
            ``retain_probability``

            The binomial function return int64 data type by
            default.  int64 multiplicated by the input
            type(floatX) always return float64.  To keep all data
            in floatX when floatX is float32, we set the dtype of
            the binomial to floatX. As in our case the value of
            the binomial is always 0 or 1, this don't change the
            result. This is needed to allow the gpu to work
            correctly as it only support float32 for now.
    """
    
    return rng.binomial(size=input.shape, n=1, p=1 - retain_probability, dtype=theano.config.floatX)

class DenoisingAutoEncoderLayer(Layer):
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

        y = s(W_encoder \tilde{x} + b)                                           (2)

        x = s(W_encoder' y  + b')                                                (3)

        L(x,z) = -sum_{k=1}^d [x_k \log z_k + (1-x_k) \log( 1-z_k)]      (4)

    """
    
    def __init__(self,
                 incoming,
                 num_units,
                 corruption_level,
                 W_encoder=init.GlorotUniform(),
                 b_encoder=init.Constant(0.),
                 b_decoder=init.Constant(0.),
                 encoder_nonlinearity=nonlinearities.sigmoid,
                 decoder_nonlinearity=nonlinearities.sigmoid,
                 W_decoder=None,
                 **kwargs):
        super(DenoisingAutoEncoderLayer, self).__init__(incoming, **kwargs)
        
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
        
        if b_encoder is None:
            self.b_encoder = None
        else:
            self.b_encoder = self.add_param(b_encoder, (num_units,), name="b_encoder",
                                    regularizable=False)
        
        if b_decoder is None:
            self.b_decoder = None
        else:
            self.b_decoder = self.add_param(b_decoder, (num_inputs,), name="b_decoder",
                                    regularizable=False)

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
        corruption_mask = get_corruption_mask(input, self.corruption_level);
        inverse_corruption_mask = 1 - corruption_mask;
        input = corruption_mask * input
        #input = self.get_corrupted_input(input);
        
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)

        return self.get_decoder_output_for(self.get_encoder_output_for(input))
    
    '''
    def get_input(self, input):
        return input
    
    def get_corrupted_input(self, input):
        corruption_mask = get_corruption_mask(input, self.corruption_level);
        return corruption_mask * input
    '''