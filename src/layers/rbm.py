import theano
import theano.tensor as T
import numpy as np

from lasagne.layers import Layer
from lasagne import init
from lasagne import nonlinearities

from theano.tensor.shared_randomstreams import RandomStreams

class RestrictedBoltzmannMachineLayer(Layer):
    """Restricted Boltzmann Machine class (rbm)
    """
    
    def __init__(self,
                 incoming,
                 num_units,
                 W=init.GlorotUniform(gain=4.0),
                 b_hidden=init.Constant(0.),
                 b_visible=init.Constant(0.),
                 **kwargs):
        super(RestrictedBoltzmannMachineLayer, self).__init__(incoming, **kwargs)
        
        self.n_hidden = num_units
        
        self.n_visible = int(np.prod(self.input_shape[1:]))
        
        self.W = self.add_param(W, (self.n_visible, self.n_hidden), name="W")
        
        if b_hidden is None:
            self.b_hidden = None
        else:
            self.b_hidden = self.add_param(b_hidden, (self.n_hidden,), name="b_hidden", regularizable=False)
        
        if b_visible is None:
            self.b_visible = None
        else:
            self.b_visible = self.add_param(b_visible, (self.n_visible,), name="b_visible", regularizable=False)
        
        '''
        # initialize input_layer layer for standalone RBM or layer0 of DBN
        self.input_layer = input_data
        if not input_data:
            self.input_layer = T.matrix('input_layer')
        '''
    
    def sample_h_given_v(self, v_sample, rng=RandomStreams()):
        ''' This function infers state of hidden units given visible units '''
        # compute the activation of the hidden units given a sample of
        # the visibles
        # pre_sigmoid_h, h_mean = self.propup(v_sample)
        
        pre_sigmoid_h = T.dot(v_sample, self.W) + self.b_hidden;
        h_mean = T.nnet.sigmoid(pre_sigmoid_h);
        
        # get a sample of the hiddens given their activation
        # Note that theano_rng.binomial returns a symbolic sample of dtype
        # int64 by default. If we want to keep our computations in floatX
        # for the GPU we need to specify to return the dtype floatX
        h_sample = rng.binomial(size=h_mean.shape, n=1, p=h_mean, dtype=theano.config.floatX)
        return [pre_sigmoid_h, h_mean, h_sample]

    def sample_v_given_h(self, h_sample, rng=RandomStreams()):
        ''' This function infers state of visible units given hidden units '''
        # compute the activation of the visible given the hidden sample
        # pre_sigmoid_v, v_mean = self.propdown(h_sample)
        
        pre_sigmoid_v = T.dot(h_sample, self.W.T) + self.b_visible
        v_mean = T.nnet.sigmoid(pre_sigmoid_v)

        # get a sample of the visible given their activation
        # Note that theano_rng.binomial returns a symbolic sample of dtype
        # int64 by default. If we want to keep our computations in floatX
        # for the GPU we need to specify to return the dtype floatX
        v_sample = rng.binomial(size=v_mean.shape, n=1, p=v_mean, dtype=theano.config.floatX)
        return [pre_sigmoid_v, v_mean, v_sample]

    def gibbs_hvh(self, h0_sample):
        ''' This function implements one step of Gibbs sampling,
            starting from the hidden state'''
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return [pre_sigmoid_v1, v1_mean, v1_sample,
                pre_sigmoid_h1, h1_mean, h1_sample]

    def gibbs_vhv(self, v0_sample):
        ''' This function implements one step of Gibbs sampling,
            starting from the visible state'''
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
        return [pre_sigmoid_h1, h1_mean, h1_sample,
                pre_sigmoid_v1, v1_mean, v1_sample]

    def get_visible_shape_for(self, input_shape):
        return input_shape
    
    def get_hidden_shape_for(self, input_shape):
        return (input_shape[0], self.n_hidden)
    
    def get_output_shape_for(self, input_shape):
        return self.get_visible_shape_for(input_shape);
    
    def get_hidden_from_visible(self, input):
        """
        Computes the encoder output given the input
        """
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)

        return self.sample_h_given_v(self, input)[-1];
    
    def get_visible_from_hidden(self, input):
        """
        Computes the encoder output given the input
        """
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)
            
        return self.sample_v_given_h(self, input)[-1];
    
    def free_energy(self, v_sample):
        ''' Function to compute the free energy '''
        wx_b = T.dot(v_sample, self.W) + self.b_hidden
        vbias_term = T.dot(v_sample, self.b_visible)
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        return -hidden_term - vbias_term

    '''
    def get_cost_updates(self, learning_rate=0.1, persistent=None, k=1):
        """This functions implements one step of CD-k or PCD-k

        :param learning_rate: learning rate used to train the RBM

        :param persistent: None for CD. For PCD, shared variable
            containing old state of Gibbs chain. This must be a shared
            variable of size (batch size, number of hidden units).

        :param k: number of Gibbs steps to do in CD-k/PCD-k

        Returns a proxy for the cost and the updates dictionary. The
        dictionary contains the update rules for weights and biases but
        also an update of the shared variable used to store the persistent
        chain, if one is used.

        """

        # decide how to initialize persistent chain:
        # for CD, we use the newly generate hidden sample
        # for PCD, we initialize from the old state of the chain
        if persistent is None:
            # compute positive phase
            pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(self.input_layer)
        
            chain_start = ph_sample
        else:
            chain_start = persistent
        # end-snippet-2
        # perform actual negative phase
        # in order to implement CD-k/PCD-k we need to scan over the
        # function that implements one gibbs step k times.
        # Read Theano tutorial on scan for more information :
        # http://deeplearning.net/software/theano/library/scan.html
        # the scan will return the entire Gibbs chain
        ([pre_sigmoid_nvs, nv_means, nv_samples, pre_sigmoid_nhs, nh_means, nh_samples], updates) = theano.scan(
            self.gibbs_hvh, outputs_info=[None, None, None, None, None, chain_start], n_steps=k)
        # start-snippet-3
        # determine gradients on RBM parameters
        # note that we only need the sample at the end of the chain
        chain_end = nv_samples[-1]
        
        cost = T.mean(self.free_energy(self.input_layer)) - T.mean(self.free_energy(chain_end))
        
        trainable_params = self.get_params(trainable=True)

        # We must not compute the gradient through the gibbs sampling
        # gparams = T.grad(cost, self.params, consider_constant=[chain_end])
        gparams = T.grad(cost, trainable_params, consider_constant=[chain_end])
        # end-snippet-3 start-snippet-4
        # constructs the update dictionary
        # for gparam, param in zip(gparams, self.params):
        for gparam, param in zip(gparams, trainable_params):
            # make sure that the learning rate is of the right dtype
            updates[param] = param - gparam * T.cast(
                learning_rate,
                dtype=theano.config.floatX
            )
        if persistent:
            # Note that this works only if persistent is a shared variable
            updates[persistent] = nh_samples[-1]
            # pseudo-likelihood is a better proxy for PCD
            monitoring_cost = self.get_pseudo_likelihood_cost(updates)
        else:
            # reconstruction cross-entropy is a better proxy for CD
            monitoring_cost = self.get_reconstruction_cost(updates, pre_sigmoid_nvs[-1])

        return monitoring_cost, updates
        # end-snippet-4

    def get_pseudo_likelihood_cost(self, updates):
        """Stochastic approximation to the pseudo-likelihood"""

        # index of bit i in expression p(x_i | x_{\i})
        bit_i_idx = theano.shared(value=0, name='bit_i_idx')

        # binarize the input_layer image by rounding to nearest integer
        xi = T.round(self.input_layer)

        # calculate free energy for the given bit configuration
        fe_xi = self.free_energy(xi)

        # flip bit x_i of matrix xi and preserve all other bits x_{\i}
        # Equivalent to xi[:,bit_i_idx] = 1-xi[:, bit_i_idx], but assigns
        # the result to xi_flip, instead of working in place on xi.
        xi_flip = T.set_subtensor(xi[:, bit_i_idx], 1 - xi[:, bit_i_idx])

        # calculate free energy with bit flipped
        fe_xi_flip = self.free_energy(xi_flip)

        # equivalent to e^(-FE(x_i)) / (e^(-FE(x_i)) + e^(-FE(x_{\i})))
        cost = T.mean(self.n_visible * T.log(T.nnet.sigmoid(fe_xi_flip - fe_xi)))

        # increment bit_i_idx % number as part of updates
        updates[bit_i_idx] = (bit_i_idx + 1) % self.n_visible

        return cost
    
    def get_reconstruction_cost(self, updates, pre_sigmoid_nv):
        """Approximation to the reconstruction error

        Note that this function requires the pre-sigmoid activation as
        input_layer.  To understand why this is so you need to understand a
        bit about how Theano works. Whenever you compile a Theano
        function, the computational graph that you pass as input_layer gets
        optimized for speed and stability.  This is done by changing
        several parts of the subgraphs with others.  One such
        optimization expresses terms of the form log(sigmoid(x)) in
        terms of softplus.  We need this optimization for the
        cross-entropy since sigmoid of numbers larger than 30. (or
        even less then that) turn to 1. and numbers smaller than
        -30. turn to 0 which in terms will force theano to compute
        log(0) and therefore we will get either -inf or NaN as
        cost. If the value is expressed in terms of softplus we do not
        get this undesirable behaviour. This optimization usually
        works fine, but here we have a special case. The sigmoid is
        applied inside the scan op, while the log is
        outside. Therefore Theano will only see log(scan(..)) instead
        of log(sigmoid(..)) and will not apply the wanted
        optimization. We can not go and replace the sigmoid in scan
        with something else also, because this only needs to be done
        on the last step. Therefore the easiest and more efficient way
        is to get also the pre-sigmoid activation as an output of
        scan, and apply both the log and sigmoid outside scan such
        that Theano can catch and optimize the expression.
        """

        cross_entropy = T.mean(T.sum(self.input_layer * T.log(T.nnet.sigmoid(pre_sigmoid_nv)) 
                                     + (1 - self.input_layer) * T.log(1 - T.nnet.sigmoid(pre_sigmoid_nv)), axis=1))
        
        return cross_entropy
    '''