import theano
import theano.tensor as T
import numpy

import sys
import lasagne

import theano.sandbox
import theano.sandbox.cuda

from lasagne.layers import Layer

from lasagne.random import get_rng

from theano.sandbox.rng_mrg import GPU_mrg_uniform, mrg_uniform

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

__all__ = [
    "GeneralizedDropoutLayer",
]

def sample_activation_probability(input_size, activation_style, activation_parameter):
    activation_probability = None;
    if activation_style == "bernoulli":
        activation_probability = numpy.zeros(input_size) + activation_parameter;
    elif activation_style == "beta_bernoulli":
        shape_alpha, shape_beta = activation_parameter;
        activation_probability = numpy.random.beta(shape_alpha, shape_beta, size=input_size);
    elif activation_style == "reciprocal_beta_bernoulli":
        shape_alpha, shape_beta = activation_parameter;
        ranked_shape_alpha = shape_alpha / numpy.arange(1, input_size + 1);
        activation_probability = numpy.zeros(input_size);
        for index in xrange(input_size):
            activation_probability[index] = numpy.random.beta(ranked_shape_alpha[index], shape_beta);
    elif activation_style == "reverse_reciprocal_beta_bernoulli":
        shape_alpha, shape_beta = activation_parameter;
        ranked_shape_alpha = shape_alpha / numpy.arange(1, input_size + 1)[::-1]; 
        activation_probability = numpy.zeros(input_size);
        for index in xrange(input_size):
            activation_probability[index] = numpy.random.beta(ranked_shape_alpha[index], shape_beta);
    elif activation_style == "mixed_beta_bernoulli":
        beta_mean, shape_beta = activation_parameter;
        scale = beta_mean / (1. - beta_mean);
        activation_probability = numpy.zeros(input_size);
        for index in xrange(input_size):
            rank = index + 1;
            activation_probability[index] = numpy.random.beta(rank * scale / shape_beta, rank / shape_beta);
    elif activation_style == "geometric":
        activation_probability = numpy.zeros(input_size);
        for index in xrange(input_size):
            rank = index + 1;
            activation_probability[index] = (activation_parameter - 1) / numpy.log(activation_parameter) * (activation_parameter ** rank)
        activation_probability = numpy.clip(activation_probability, 0., 1.);
    elif activation_style == "reciprocal":
        activation_probability = activation_parameter / numpy.arange(1, input_size + 1);
        activation_probability = numpy.clip(activation_probability, 0., 1.);
    elif activation_style == "exponential":
        activation_probability = activation_parameter / numpy.arange(1, input_size + 1);
        activation_probability = numpy.clip(activation_probability, 0., 1.);
    else:
        sys.stderr.write("error: unrecognized configuration...\n");
        sys.exit();

    return activation_probability.astype(numpy.float32)

def get_filter(input_shape, retain_probability, rng=RandomStreams()):
    filter = rng.binomial(size=input_shape, n=1, p=retain_probability, dtype=theano.config.floatX);
    '''
    if isinstance(input_shape, tuple):
        filter = numpy.zeros(input_shape);
        for dim in xrange(len(retain_probability)):
            filter[:, dim] = self._srng.binomial(size=len(input_shape[0]), n=1, p=retain_probability[dim], dtype=theano.config.floatX)
    else:
        if isinstance(input_shape, T.Variable) and input_shape.ndim == 1:
            #filter = rng.binomial(size=input_shape, n=1, p=0.5, dtype=theano.config.floatX);
            filter = self._srng.normal(size=input_shape, avg=0.0, std=1.0, dtype=theano.config.floatX);
    '''
    
    return filter

class GeneralizedDropoutLayer(Layer):
    """Dropout layer

    Sets values to zero with probability activation_probability. See notes for disabling dropout
    during testing.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        the layer feeding into this layer, or the expected input shape
    activation_probability : float or scalar tensor
        The probability of setting a value to 1
    rescale : bool
        If true the input is rescaled with input / activation_probability when deterministic
        is False.

    Notes
    -----
    The dropout layer is a regularizer that randomly sets input values to
    zero; see [1]_, [2]_ for why this might improve generalization.
    During training you should set deterministic to false and during
    testing you should set deterministic to true.

    If rescale is true the input is scaled with input / activation_probability when
    deterministic is false, see references for further discussion. Note that
    this implementation scales the input at training time.

    References
    ----------
    .. [1] Hinton, G., Srivastava, N., Krizhevsky, A., Sutskever, I.,
           Salakhutdinov, R. R. (2012):
           Improving neural networks by preventing co-adaptation of feature
           detectors. arXiv preprint arXiv:1207.0580.

    .. [2] Srivastava Nitish, Hinton, G., Krizhevsky, A., Sutskever,
           I., & Salakhutdinov, R. R. (2014):
           Dropout: A Simple Way to Prevent Neural Networks from Overfitting.
           Journal of Machine Learning Research, 5(Jun)(2), 1929-1958.
    """
    def __init__(self, incoming, activation_probability, rescale=True, **kwargs):
        super(GeneralizedDropoutLayer, self).__init__(incoming, **kwargs)
        self._srng = RandomStreams(get_rng().randint(1, 2147462579))
        
        '''
        self._alpha_alpha = alpha;
        
        assert len(self.input_shape)==2;
        dimensionality = self.input_shape[1];
        #dimensionality = np.prod(self.input_shape[1:]);
        
        shape_alpha = self._alpha_alpha / numpy.arange(1, dimensionality + 1);
        shape_beta = 1.0;
        
        activation_probability = numpy.zeros(dimensionality);
        for index in xrange(dimensionality):
            activation_probability[index] = numpy.random.beta(shape_alpha[index], shape_beta);
        '''
        
        self.activation_probability = activation_probability;
        
        self.rescale = rescale

    def get_output_for(self, input, deterministic=False, **kwargs):
        """
        Parameters
        ----------
        input : tensor
            output from the previous layer
        deterministic : bool
            If true dropout and scaling is disabled, see notes
        """
        if deterministic or numpy.all(self.activation_probability == 1):
            return input
        else:
            retain_prob = self.activation_probability
            if self.rescale:
                input /= retain_prob
                
            # use nonsymbolic shape for dropout mask if possible
            input_shape = self.input_shape
            if any(s is None for s in input_shape):
                input_shape = input.shape
            
            return input * get_filter(input_shape, retain_prob, rng=RandomStreams())