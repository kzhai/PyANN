PyMLP
==========

PyMLP is an Multi-Layer Perceptron package.

Please download the latest version from our [GitHub repository](https://github.com/kzhai/PyMLP).

Please send any bugs of problems to Ke Zhai (kzhai@umd.edu).

Install and Build
----------

This package depends on many external python libraries, such as numpy, scipy, theano, and lasagne.

Launch and Execute
----------

Assume the PyMLP package is downloaded under directory ```$PROJECT_SPACE/src/```, i.e., 

	$PROJECT_SPACE/src/PyMLP

To prepare the example dataset,

	tar zxvf mnist.tar.gz

To launch PyMLP, first redirect to the parent directory of PyMLP source code,

	cd $PROJECT_SPACE/src/

and run the following command on example dataset,

	python -m PyMLP.launch_train --input_directory=./PyMLP/mnist/ --output_directory=./PyMLP/ --minibatch_size=100 --number_of_epochs=50 --learning_rate=0.01 --L2_regularizer_lambdas=0.0001 --layer_dimensions=784,800,500,10 --layer_nonlinearities=sigmoid,tanh,softmax

The generic argument to run PyMLP is

	python -m PyMLP.launch_train --input_directory=$INPUT_DIRECTORY/$DATASET_NAME --output_directory=$OUTPUT_DIRECTORY --number_of_epochs=$NUMBER_OF_EPOCHS --layer_dimensions=$DIM_0,$DIM_1,...,$DIM_n --layer_nonlinearities=$F_1,$F_2,...,$F_n

You should be able to find the output at directory ```$OUTPUT_DIRECTORY/$DATASET_NAME```.

Under any cirsumstances, you may also get help information and usage hints by running the following command

	python -m PyMLP.launch_train --help

To use it as a logistic regression model

	python -m PyMLP.launch_train --input_directory=$INPUT_DIRECTORY/$DATASET_NAME/ --output_directory=$OUTPUT_DIRECTORY --number_of_epochs=$NUMBER_OF_EPOCHS --layer_dimensions=$DIM_0,$DIM_1 --layer_nonlinearities=softmax
