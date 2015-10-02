PyANN
==========

PyANN is an Artificial Neural Network package.

Please download the latest version from our [GitHub repository](https://github.com/kzhai/PyANN).

Please send any bugs of problems to Ke Zhai (kzhai@umd.edu).

Install and Build
----------

This package depends on many external python libraries, such as numpy, scipy, theano, and nltk.

Launch and Execute
----------

Assume the PyANN package is downloaded under directory ```$PROJECT_SPACE/src/```, i.e., 

	$PROJECT_SPACE/src/PyANN

To prepare the example dataset,

	tar zxvf mnist.tar.gz

To launch PyANN, first redirect to the parent directory of PyANN source code,

	cd $PROJECT_SPACE/src/

and run the following command on example dataset,

	python -m PyANN.launch_train --input_directory=./PyANN/mnist/ --output_directory=./PyANN/ --minibatch_size=10 --number_of_epochs=500 --learning_rate=0.01 --L2_regularizer_lambda=0.0001 --dimensionalities=784,500,10 --activation_functions=sigmoid,softmax

The generic argument to run PyANN is

	python -m PyANN.launch_train --input_directory=$INPUT_DIRECTORY/$DATASET_NAME --output_directory=$OUTPUT_DIRECTORY --minibatch_size=$MINIBATCH_SIZE --number_of_epochs=$NUMBER_OF_EPOCHS --learning_rate=$LEARNING_RATE --dimensionalities=$DIM_1,$DIM_2,...,$DIM_n --activation_functions=$F_1,$F_2,...,$F_n

You should be able to find the output at directory ```$OUTPUT_DIRECTORY/$DATASET_NAME```.

Under any cirsumstances, you may also get help information and usage hints by running the following command

	python -m PyANN.launch_train --help

To use it as a logistic regression model

	python -m PyANN.launch_train --input_directory=$INPUT_DIRECTORY/$DATASET_NAME/ --output_directory=$OUTPUT_DIRECTORY --minibatch_size=$MINIBATCH_SIZE --number_of_epochs=$NUMBER_OF_EPOCHS --learning_rate=$LEARNING_RATE --dimensionalities=$DIM_IN,$DIM_OUT --activation_functions=softmax
