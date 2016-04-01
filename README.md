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

Assume the PyANN package is downloaded under directory ```$PROJECT_SPACE/```, i.e., 

	$PROJECT_SPACE/PyANN

To prepare the example dataset,

	cd $PROJECT_SPACE/PyANN/input
	tar zxvf mnist.tar.gz

To launch PyANN modules, first redirect to the parent directory of PyANN source code,

	cd $PROJECT_SPACE/PyANN/src

### Launch multi-layer perceptron (MLP)

To launch multi-layer perceptron (MLP) on mnist example dataset,

	python -um PyMLP.launch_train \
		--input_directory=../input/mnist_784/ \
		--output_directory=../output/ \
		--minibatch_size=1 \
		--number_of_epochs=1000 \
		--learning_rate=0.001 \
		--number_of_training_data=50000 \
		--objective_to_minimize=categorical_crossentropy \
		--layer_dimensions=784,1024,1024,10 \
		--layer_nonlinearities=sigmoid,sigmoid,softmax

The generic argument to run MLP is

	python -um PyMLP.launch_train \
		--input_directory=$INPUT_DIRECTORY/$DATASET_NAME \
		--output_directory=$OUTPUT_DIRECTORY/ \
		--minibatch_size=$MINI_BATCH_SIZE \
		--number_of_epochs=$NUMBER_OF_EPOCHS \
	  	--learning_rate=$LEARNING_RATE \
		--number_of_training_data=$NUMBER_OF_TRAINING_DATA \
		--objective_to_minimize=$OBJECTIVE_TO_MINIMIZE \
		--snapshot_interval=$SNAPSHOT_INTERVAL \
		--layer_dimensions=$DIM_1,$DIM_2,...,$DIM_n \
		--layer_nonlinearities=$F_1,$F_2,...,$F_n

To use it as a logistic regression model

	python -um PyMLP.launch_train \
		--input_directory=$INPUT_DIRECTORY/$DATASET_NAME/ \
		--output_directory=$OUTPUT_DIRECTORY \
		--minibatch_size=$MINI_BATCH_SIZE \
		--number_of_epochs=$NUMBER_OF_EPOCHS \
		--learning_rate=$LEARNING_RATE \
		--dimensionalities=$DIM_IN,$DIM_OUT \
		--activation_functions=softmax

Under any cirsumstances, you may also get help information and usage hints by running the following command

	python -um PyMLP.launch_train --help

### Launch convolutional neural network (CNN)

Model Output and Snapshot
----------

You should be able to find the output at directory ```$OUTPUT_DIRECTORY/$DATASET_NAME```.
