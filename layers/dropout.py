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
        The probability of setting a value to zero
    rescale : bool
        If true the input is rescaled with input / (1-activation_probability) when deterministic
        is False.

    Notes
    -----
    The dropout layer is a regularizer that randomly sets input values to
    zero; see [1]_, [2]_ for why this might improve generalization.
    During training you should set deterministic to false and during
    testing you should set deterministic to true.

    If rescale is true the input is scaled with input / (1-activation_probability) when
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
        if deterministic or numpy.all(self.activation_probability == 0):
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

class temp():
    def uniform(self, size, low=0.0, high=1.0, ndim=None, dtype=None,
                nstreams=None):
        """
        Sample a tensor of given size whose element from a uniform
        distribution between low and high.
        If the size argument is ambiguous on the number of dimensions,
        ndim may be a plain integer to supplement the missing information.
        Parameters
        ----------
        low
            Lower bound of the interval on which values are sampled. 
            If the ``dtype`` arg is provided, ``low`` will be cast into
            dtype. This bound is excluded.
        high
            Higher bound of the interval on which values are sampled.
            If the ``dtype`` arg is provided, ``high`` will be cast into
            dtype. This bound is excluded.
        size
          Can be a list of integer or Theano variable (ex: the shape
          of other Theano Variable).
        dtype
            The output data type. If dtype is not specified, it will be
            inferred from the dtype of low and high, but will be at
            least as precise as floatX.
        """
        low = T.as_tensor_variable(low)
        high = T.as_tensor_variable(high)
        if dtype is None:
            dtype = T.scal.upcast(T.config.floatX, low.dtype, high.dtype)

        low = T.cast(low, dtype=dtype)
        high = T.cast(high, dtype=dtype)

        if isinstance(size, tuple):
            msg = "size must be a tuple of int or a Theano variable"
            assert all([isinstance(i, (numpy.integer, int, T.Variable))
                        for i in size]), msg
            if any([isinstance(i, (numpy.integer, int)) and i <= 0
                    for i in size]):
                raise ValueError(
                    "The specified size contains a dimension with value <= 0",
                    size)
        else:
            if not (isinstance(size, T.Variable) and size.ndim == 1):
                raise TypeError("size must be a tuple of int or a Theano "
                                "Variable with 1 dimension, got " + str(size) +
                                " of type " + str(type(size)))
        orig_nstreams = nstreams
        if nstreams is None:
            nstreams = self.n_streams(size)
        rstates = self.get_substream_rstates(nstreams, dtype)

        if self.use_cuda and dtype == 'float32':
            node_rstate = theano.sandbox.cuda.float32_shared_constructor(rstates)
            assert isinstance(node_rstate.type, theano.sandbox.cuda.CudaNdarrayType)

            # we can't use the normal mrg_uniform constructor + later
            # optimization
            # because of the tmp_float_buf hack above.  There is
            # currently no Theano node that will do a frombuffer
            # reinterpretation.
            u = self.pretty_return(node_rstate,
                                   *GPU_mrg_uniform.new(node_rstate,
                                                        ndim, dtype, size),
                                   size=size, nstreams=orig_nstreams)
        else:
            node_rstate = theano.shared(rstates)
            u = self.pretty_return(node_rstate,
                                   *mrg_uniform.new(node_rstate,
                                                    ndim, dtype, size),
                                   size=size, nstreams=orig_nstreams)
        # Add a reference to distinguish from other shared variables
        node_rstate.tag.is_rng = True
        r = u * (high - low) + low

        if u.type.broadcastable != r.type.broadcastable:
            raise NotImplementedError(
                'Increase the size to match the broadcasting pattern of '
                '`low` and `high` arguments')

        assert r.dtype == dtype
        return r
        
    def binomial(self, size=None, n=1, p=None, ndim=None, dtype='int64', nstreams=None):
        if n == 1:
            if dtype == 'float32' and self._srng.use_cuda:
                x = self._srng.uniform(size=size, dtype=dtype, nstreams=nstreams)
            else:
                x = self._srng.uniform(size=size, nstreams=nstreams)
            return T.cast(x < p, dtype)
        else:
            raise NotImplementedError("MRG_RandomStreams.binomial with n > 1")
