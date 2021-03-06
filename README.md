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
		--input_directory=../input/mnist_784_unitized/ \
		--output_directory=../output/ \
		--minibatch_size=1 \
		--number_of_epochs=1000 \
		--learning_rate=0.001 \
		--number_of_training_data=50000 \
		--objective_to_minimize=categorical_crossentropy \
		--layer_dimensions=1024,1024,10 \
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
		--layer_dimensions=$DIM_1,...,$DIM_n \
		--layer_nonlinearities=$F_1,$F_2,...,$F_n

To use it as a logistic regression model

	python -um PyMLP.launch_train \
		--input_directory=$INPUT_DIRECTORY/$DATASET_NAME/ \
		--output_directory=$OUTPUT_DIRECTORY \
		--minibatch_size=$MINI_BATCH_SIZE \
		--number_of_epochs=$NUMBER_OF_EPOCHS \
		--learning_rate=$LEARNING_RATE \
		--layer_dimensions=$DIM_OUT \
		--layer_nonlinearities=softmax

Under any cirsumstances, you may also get help information and usage hints by running the following command

	python -um PyMLP.launch_train --help

### Launch convolutional neural network (CNN)

To launch convolutional neural network (CNN) on mnist example dataset,

	python -um PyCNN.launch_train \
		--input_directory=../input/mnist_1x28x28_unitized/ \
		--output_directory=../output/ \
		--minibatch_size=1 \
		--number_of_epochs=1000 \
		--learning_rate=0.001 \
		--number_of_training_data=50000 \
		--objective_to_minimize=categorical_crossentropy \
		--convolution_filter=32,64 \
		--convolution_nonlinearities=tanh,tanh \
		--dense_dimensions=1024,10 \
		--dense_nonlinearities=tanh,softmax

The generic argument to run CNN is

	python -um PyCNN.launch_train \
		--input_directory=$INPUT_DIRECTORY/$DATASET_NAME \
		--output_directory=$OUTPUT_DIRECTORY/ \
		--minibatch_size=$MINI_BATCH_SIZE \
		--number_of_epochs=$NUMBER_OF_EPOCHS \
	  	--learning_rate=$LEARNING_RATE \
		--number_of_training_data=$NUMBER_OF_TRAINING_DATA \
		--objective_to_minimize=$OBJECTIVE_TO_MINIMIZE \
		--convolution_filter=$CONV_FILTER_1,$CONV_FILTER_2,...,$CONV_FILTER_n \
		--convolution_nonlinearities=$CONV_F_1,$CONV_F_2,...,$CONV_F_n \
		--dense_dimensions=$DENSE_DIM_1,$DENSE_DIM_2,...,$DENSE_DIM_m \
		--dense_nonlinearities=$DENSE_F_1,$DENSE_F_2,...,$DENSE_F_m

Under any cirsumstances, you may also get help information and usage hints by running the following command

	python -um PyCNN.launch_train --help

### Launch denoising auto-encoders (DAE)

To launch denoising auto-encoders (DAE) on mnist example dataset,

	python -um PyDAE.launch_train \
		--input_directory=../input/mnist_784_unitized/ \
		--output_directory=../output/ \
		--minibatch_size=1 \
		--number_of_epochs=15 \
		--learning_rate=0.001 \
		--objective_to_minimize=binary_crossentropy \
		--layer_dimension=1024 \
		--layer_nonlinearity=sigmoid \
		--layer_corruption_level=0.2
  
The generic argument to run DAE is

	python -um PyDAE.launch_train \
		--input_directory=$INPUT_DIRECTORY/$DATASET_NAME \
		--output_directory=$OUTPUT_DIRECTORY/ \
		--minibatch_size=$MINI_BATCH_SIZE \
		--number_of_epochs=$NUMBER_OF_EPOCHS \
	  	--learning_rate=$LEARNING_RATE \
		--objective_to_minimize=$OBJECTIVE_TO_MINIMIZE \
		--layer_dimension=$DIM \
		--layer_nonlinearity=$F \
		--layer_corruption_level=$CORR

Under any cirsumstances, you may also get help information and usage hints by running the following command

	python -um PyDAE.launch_train --help

### Launch stacked denoising auto-encoders (SDAE)

To launch stacked denoising auto-encoders (SDAE) on mnist example dataset,

	python -um PySDAE.launch_train \
		--input_directory=../input/mnist_784_unitized/ \
		--output_directory=../output/ \
		--minibatch_size=1 \
		--number_of_epochs=15 \
		--learning_rate=0.001 \
		--objective_to_minimize=binary_crossentropy \
		--layer_dimensions=1024,1024,1024 \
		--layer_nonlinearities=sigmoid,sigmoid,sigmoid \
		--layer_corruption_levels=0.1,0.2,0.3

The generic argument to run SDAE is

	python -um PySDAE.launch_train \
		--input_directory=$INPUT_DIRECTORY/$DATASET_NAME \
		--output_directory=$OUTPUT_DIRECTORY/ \
		--minibatch_size=$MINI_BATCH_SIZE \
		--number_of_epochs=$NUMBER_OF_EPOCHS \
	  	--learning_rate=$LEARNING_RATE \
		--objective_to_minimize=$OBJECTIVE_TO_MINIMIZE \
		--layer_dimensions=$DIM_1,...,$DIM_n \
		--layer_nonlinearities=$F_1,$F_2,...,$F_n \
		--layer_corruption_levels=$CORR_1,$CORR_2,...,$CORR_n

Under any cirsumstances, you may also get help information and usage hints by running the following command

	python -um PySDAE.launch_train --help

### Launch restricted Boltzmann machines (RBM)

To launch restricted Boltzmann machines (RBM) on mnist example dataset,

	python -um PyRBM.launch_train \
		--input_directory=../input/mnist_784_unitized/ \
		--output_directory=../output/ \
		--minibatch_size=1 \
		--number_of_epochs=15 \
		--learning_rate=0.01 \
		--layer_dimension=1024 \
		--number_of_gibbs_steps=15 \
		--persistent
		
The generic argument to run RBM is

	python -um PyRBM.launch_train \
		--input_directory=$INPUT_DIRECTORY/$DATASET_NAME \
		--output_directory=$OUTPUT_DIRECTORY/ \
		--minibatch_size=$MINI_BATCH_SIZE \
		--number_of_epochs=$NUMBER_OF_EPOCHS \
	  	--learning_rate=$LEARNING_RATE \
		--layer_dimension=$DIM \
		--number_of_gibbs_steps=$NUMBER_OF_GIBBS_STEPS \
		--persistent

Under any cirsumstances, you may also get help information and usage hints by running the following command

	python -um PyRBM.launch_train --help

### Launch deep belief networks (DBN)

To launch deep belief networks (DBN) on mnist example dataset,

	python -um PyDBN.launch_train \
		--input_directory=../input/mnist_784_unitized/ \
		--output_directory=../output/ \
		--minibatch_size=1 \
		--number_of_epochs=100 \
		--learning_rate=0.01 \
		--layer_dimensions=1024,1024,1024 \
		--number_of_gibbs_steps=1 \
		--persistent
		
The generic argument to run DBN is 

	python -um PyDBN.launch_train \
		--input_directory=$INPUT_DIRECTORY/$DATASET_NAME \
		--output_directory=$OUTPUT_DIRECTORY/ \
		--minibatch_size=$MINI_BATCH_SIZE \
		--number_of_epochs=$NUMBER_OF_EPOCHS \
	  	--learning_rate=$LEARNING_RATE \
		--layer_dimensions=$DIM_1,...,$DIM_n \
		--number_of_gibbs_steps=$NUMBER_OF_GIBBS_STEPS \
		--persistent

Under any cirsumstances, you may also get help information and usage hints by running the following command

	python -um PyDBN.launch_train --help

### Launch recurrent neural network (RNN)

To launch recurrent neural network (RNN) on atis example dataset,

	python -um PyRNN.launch_train \
		--input_directory=../input/atis.fold0.label/ \
		--output_directory=../output/ \
		--number_of_epochs=50 \
		--learning_rate=0.01 \
		--embedding_dimension=100 \
		--sequence_length=9 \
		--window_size=5 \
		--number_of_training_data=3000 \
		--layer_dimensions=256,[128],127 \
		--layer_nonlinearities=sigmoid,[sigmoid],softmax \
		--objective_to_minimize=categorical_crossentropy
		
The generic argument to run RNN is 

	python -um PyRNN.launch_train \
		--input_directory=$INPUT_DIRECTORY/$DATASET_NAME \
		--output_directory=$OUTPUT_DIRECTORY/ \
		--number_of_epochs=$NUMBER_OF_EPOCHS \
	  	--learning_rate=$LEARNING_RATE \
	  	--embedding_dimension=$EMBEDDING_DIMENSION \
		--sequence_length=$SEQUENCE_LENGTH \
		--window_size=$WINDOW_SIZE \
		--layer_dimensions=$DIM_1,...,[$RNN_DIM_1,...,$RNN_DIM_N],...,$DIM_N,... \
		--layer_nonlinearities=$F_1,...,[$RNN_F_1,...,$RNN_F_N],...,$F_N,... \
		--objective_to_minimize=categorical_crossentropy
		
Under any circumstances, you may also get help information and usage hints by running the following command

	python -um PyRNN.launch_train --help

Note that sometimes you may get following value casting error

	ValueError: When compiling the inner function of scan the following error has been encountered: The initial state (`outputs_info` in scan nomenclature) of variable IncSubtensor{Set;:int64:}.0 (argument number 2) has dtype float32, while the result of the inner function (`fn`) has dtype float64. This can happen if the inner function of scan results in an upcast or downcast.

I do not have a clear understanding on why such error is raised during ```scan``` function, likely resulted from an internal type casting operation. This is possibility due to some precision constrain in regards to the well-known gradient vanishing and exploding problem.
In such case, you may want to include following ```THEANO_FLAGS``` when invoking the module (just insert the following command before the python command).

	THEANO_FLAGS='floatX=float64'

### Launch connectionist temporal classification (CTC)

To launch connectionist temporal classification (CTC) on atis example dataset,

	python -m PyCTC.launch_train \
		--input_directory=../input/atis.fold0.label/ \
		--output_directory=../output/ \
		--number_of_epochs=50 \
		--learning_rate=0.01 \
		--embedding_dimension=100 \
		--sequence_length=100 \
		--number_of_training_data=3000 \
		--layer_dimensions=256,[128],127 \
		--layer_nonlinearities=sigmoid,[sigmoid],softmax \
		--objective_to_minimize=categorical_crossentropy

The generic argument to run CTC is

	python -um PyCTC.launch_train \
		--input_directory=$INPUT_DIRECTORY/$DATASET_NAME \
		--output_directory=$OUTPUT_DIRECTORY/ \
		--number_of_epochs=$NUMBER_OF_EPOCHS \
	  	--learning_rate=$LEARNING_RATE \
	  	--embedding_dimension=$EMBEDDING_DIMENSION \
		--sequence_length=$SEQUENCE_LENGTH \
		--window_size=$WINDOW_SIZE \
		--layer_dimensions=$DIM_1,...,[$RNN_DIM_1,...,$RNN_DIM_N],...,$DIM_N,... \
		--layer_nonlinearities=$F_1,...,[$RNN_F_1,...,$RNN_F_N],...,$F_N,... \
		--objective_to_minimize=categorical_crossentropy

Under any cirsumstances, you may also get help information and usage hints by running the following command

	python -um PyCTC.launch_train --help

Model Output and Snapshot
----------

You should be able to find the output at directory ```$OUTPUT_DIRECTORY/$DATASET_NAME```.
