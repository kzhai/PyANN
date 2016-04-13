import theano
import theano.tensor as T
import numpy

from lasagne.layers import Layer
from lasagne import init, nonlinearities
from lasagne import nonlinearities

import lasagne

import network

from layers.rbm import RestrictedBoltzmannMachineLayer

from theano.tensor.shared_randomstreams import RandomStreams

class RestrictedBoltzmannMachine(network.Network):
    def __init__(self,
            input_network,
            layer_dimension,
            # encoder_nonlinearity=lasagne.nonlinearities.sigmoid,
            # decoder_nonlinearity=lasagne.nonlinearities.identity,
            # objective_to_minimize="free_energy",
            # corruption_level=0,
            W=init.GlorotUniform(gain=4.0),
            b_hidden=init.Constant(0.),
            b_visible=init.Constant(0.),
            # L1_regularizer_lambdas=None,
            # L2_regularizer_lambdas=None,
            **kwargs):
        super(RestrictedBoltzmannMachine, self).__init__(input_network)
        
        self.input = lasagne.layers.get_output(self.input_network);

        self.input_shape = int(numpy.prod(lasagne.layers.get_output_shape(input_network)[1:]))
        
        network = input_network;
        network = RestrictedBoltzmannMachineLayer(
            network,
            layer_dimension,
            W=W,
            b_hidden=b_hidden,
            b_visible=b_visible,
            );
        
        self.network = network;
        
        # self.set_L1_regularizer_lambda(L1_regularizer_lambdas);
        # self.set_L2_regularizer_lambda(L2_regularizer_lambdas);
            
        # assert objective_to_minimize != None;
        # self.objective_to_minimize = network.free_energy();
    
    def free_energy(self, input=None):
        if input == None:
            return self.network.free_energy(self.input);
        else:
            return self.network.free_energy(input);
        
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
            pre_sigmoid_ph, ph_mean, ph_sample = self.network.sample_h_given_v(self.input)
        
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
            self.network.gibbs_hvh, outputs_info=[None, None, None, None, None, chain_start], n_steps=k)
        # start-snippet-3
        # determine gradients on RBM parameters
        # note that we only need the sample at the end of the chain
        chain_end = nv_samples[-1]
        
        cost = T.mean(self.free_energy(self.input)) - T.mean(self.free_energy(chain_end))
        
        trainable_params = self.get_all_params(trainable=True)

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

        return monitoring_cost, chain_end, updates
        # end-snippet-4

    def get_pseudo_likelihood_cost(self, updates):
        """Stochastic approximation to the pseudo-likelihood"""

        # index of bit i in expression p(x_i | x_{\i})
        bit_i_idx = theano.shared(value=0, name='bit_i_idx')

        # binarize the input_layer image by rounding to nearest integer
        xi = T.round(self.input)

        # calculate free energy for the given bit configuration
        fe_xi = self.free_energy(xi)

        # flip bit x_i of matrix xi and preserve all other bits x_{\i}
        # Equivalent to xi[:,bit_i_idx] = 1-xi[:, bit_i_idx], but assigns
        # the result to xi_flip, instead of working in place on xi.
        xi_flip = T.set_subtensor(xi[:, bit_i_idx], 1 - xi[:, bit_i_idx])

        # calculate free energy with bit flipped
        fe_xi_flip = self.free_energy(xi_flip)

        # equivalent to e^(-FE(x_i)) / (e^(-FE(x_i)) + e^(-FE(x_{\i})))
        cost = T.mean(self.input_shape * T.log(T.nnet.sigmoid(fe_xi_flip - fe_xi)))

        # increment bit_i_idx % number as part of updates
        updates[bit_i_idx] = (bit_i_idx + 1) % self.input_shape

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

        cross_entropy = T.mean(T.sum(self.input * T.log(T.nnet.sigmoid(pre_sigmoid_nv)) 
                                     + (1 - self.input) * T.log(1 - T.nnet.sigmoid(pre_sigmoid_nv)), axis=1))
        
        return cross_entropy
    
    # def get_objective_to_minimize(self, k=1):
        # return self.network.free_energy();
