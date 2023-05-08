import numpy as np

import tensorflow as tf
# enables eager execution for TF 1.x (TF 2 has it as default)
# tf.enable_eager_execution()
# disable eager mode for TF 2
# tf.compat.v1.disable_eager_execution()

from tensorflow.keras.layers import Input, Dense, Activation
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

import matplotlib
matplotlib.use('pdf')  # noqa
# display figures in the notebook
import matplotlib.pyplot as plt
import argparse
import seaborn as sns

import utils.cifar10 as cifar10


SEED_NUM = 16823
np.random.seed(SEED_NUM)  # for reproducibility 

PATH_MNIST = '/p/project/training2306/datasets/MNIST/mnist.npz'
PATH_CIFAR10 = '/p/project/training2306/datasets/cifar10'


def load_MNIST(path):

    with np.load(path, allow_pickle=True) as f:
        X_train_MNIST, Y_train_MNIST = f['x_train'], f['y_train']
        X_test_MNIST, Y_test_MNIST = f['x_test'], f['y_test']

    return (X_train_MNIST, Y_train_MNIST), (X_test_MNIST, Y_test_MNIST)


def plot_data(X, y):
   
    fig, axes = plt.subplots(ncols=5, nrows=3, figsize=(12, 9))

    for i, ax in enumerate(axes.ravel()):
        ax.imshow((X[i]),cmap=plt.cm.gray_r, interpolation='nearest')
        ax.set_title("training label: %d"% y[i])

    fig.savefig('input_data_plot' + ".pdf")
        

def plot_label_distribution(data):
    unique_train, counts_train = np.unique(data, return_counts=True)

    fig, ax0 = plt.subplots(ncols=1, sharex=True, figsize=(15, 6))

    ax0.bar(unique_train, counts_train, color='darkgreen')
    ax0.set_xticks(unique_train)
    ax0.set_xlabel("Label")
    ax0.set_ylabel("Quantity")
    ax0.set_title("Labels in MNIST training dataset")

    fig.savefig('plot_label_distribution' + ".pdf")


def plot_accuracy(history, plot_filename):
    import pandas as pd

    fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, figsize=(12, 6))
    history_df = pd.DataFrame(history.history)
    history_df["epoch"] = history.epoch

    # For TF 1.x 'acc' and 'val_acc'
    history_df.plot(x="epoch", y=["loss", "val_loss"], ax=ax0)
    history_df.plot(x="epoch", y=["accuracy", "val_accuracy"], ax=ax1)

    filename = plot_filename
    fig.savefig(filename + ".pdf")


def mnist_fit_FCNet_model_on_unnormalised_data(learning_rate, epochs=5,batch_size=32, nesterov=False, momentum=0.0):

    # Load data and preprocess
    (X_train_MNIST, Y_train_MNIST), (X_test_MNIST, Y_test_MNIST) = load_MNIST(PATH_MNIST) 
    # load MNIST data

    input_channels = 1

    # dataset properties
    train_samples = X_train_MNIST.shape[0]
    test_samples = X_test_MNIST.shape[0]
    input_height = X_train_MNIST.shape[1]
    input_width = X_train_MNIST.shape[2]

    # flatten images for input to a fully connected dense network
    X_train = X_train_MNIST.reshape(train_samples, input_height * input_width, input_channels)
    X_test = X_test_MNIST.reshape(test_samples, input_height * input_width, input_channels)

    # Convert training labels to categorical data (0 ... 1 .... 0), one-hot - to provide proper form to compute loss
    Y_train = to_categorical(Y_train_MNIST)
    # Test labels for trained model testing on the test set
    Y_test = Y_test_MNIST

    input_dim = X_train.shape[1]
    
    # Shape of input to the network: spatial dimensions (here 1 dim flat vector), and channels number 
    # (e.g 3 for RGB, here 1)
    input_shape = (input_dim, input_channels)

    hidden_dim = 128
    output_dim = 10
    activation_func = 'relu' # sigmoid, tanh, relu, elu, ...

    # Define a set for validation: 5000 out of 60000 - 55000 train, 5000 validation
    validation_split = 1/12 

    # -------- Network Barebone Description, Functional Keras API --------------

    inputs = Input(shape = (input_dim, ))
    x = Dense(hidden_dim, activation = activation_func, input_shape = input_shape)(inputs)
    output = Dense(output_dim, activation='softmax')(x)

    # -------- Compile network model  --------------
    model = Model(inputs, output)


    optimizer=optimizers.SGD(lr=learning_rate, nesterov=nesterov, momentum=momentum)
    model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # print out network description
    model.summary()

    history = model.fit(X_train, Y_train, validation_split=validation_split, epochs=epochs, batch_size=batch_size)

    return history 



def mnist_fit_FCNet_model(learning_rate=0.01, epochs=5, batch_size=32, nesterov=False, momentum=0.0):

    (X_train_MNIST, Y_train_MNIST), (X_test_MNIST, Y_test_MNIST) = load_MNIST(PATH_MNIST) # load MNIST data

    # dataset properties
    train_samples = X_train_MNIST.shape[0]
    test_samples = X_test_MNIST.shape[0]
    input_height = X_train_MNIST.shape[1]
    input_width = X_train_MNIST.shape[2]

    # one channel only for MNIST input to the network
    input_channels = 1

    print("Labels in the original data:", Y_train_MNIST[0:10])

    # flatten images for input to a fully connected dense network
    X_train = X_train_MNIST.reshape(train_samples, input_height * input_width, input_channels)
    X_test = X_test_MNIST.reshape(test_samples, input_height * input_width, input_channels)

    # Convert training labels to categorical data (0 ... 1 .... 0), one-hot - to provide proper form to compute loss
    Y_train = to_categorical(Y_train_MNIST)
    # Test labels for trained model testing on the test set
    Y_test = Y_test_MNIST

    # --- INFO print
    print("Input shape train data original:", X_train_MNIST.shape[1:3])
    print("Input shape flat:", X_train.shape[1])
    print("Label shape (one-hot):", Y_train.shape[1])
    print("Labels one-hot examples: \n", Y_train[0:2])

    # --- properties of network input and output format
    input_dims = X_train.shape[1]
    num_classes = Y_train.shape[1]
    print("Number of classes: ", num_classes)


    # --- Preprocessing and normalizing input

    # --- compute training set stats

    max_value = X_train.max()
    min_value = X_train.min()

    mean_value = X_train.mean()
    std_value = X_train.std()

    print("Max value train: ", max_value)
    print("Min value train: ", min_value)

    print("Mean value train: ", mean_value)
    print("Std value train: ", std_value)

    # --- applying scaling factor for train and test
    rescaling_max = max_value

    # --- rescale input
    X_train = X_train.astype('float32') / rescaling_max  # Normalize between 0 and 1
    # ! USE ON TEST only TRAIN normalizing factors !
    # Otherwise considered as sneaking information from train to test
    X_test = X_test.astype('float32') / rescaling_max


    # ---- zero mean, unit variance

    # mean = 0 ; standard deviation = 1.0

    mean_train = X_train.mean()
    std_train = X_train.std()
    X_train = (X_train - mean_train)/(std_train + 0.1)
    # -- transforming test using scaling parameters from X_train - trainings set !!!
    X_test = (X_test - mean_train)/(std_train + 0.1)

    SEED_NUM = 16823
    np.random.seed(SEED_NUM)  # for reproducibility

    # Shape of input to the network: spatial dimensions (here 1 dim flat vector), and channels number
    # (e.g 3 for RGB, here 1)
    input_shape = (input_dims, input_channels)


    hidden_dim = 128
    output_dim = 10
    activation_func = 'relu' # sigmoid, tanh, relu, elu, ...


    validation_split = 1/12 # 5000 out of 60000; 55000 train, 5000 validation

    # experiment with weight initialization only with deeper networks; 
    # RELU the same - the deeper the network, the less stable training; additional measures neccessary

    # -------- Network Barebone Description, Functional Keras API --------------

    inputs = Input(shape = (input_dims, ))
    x = Dense(hidden_dim, activation = activation_func, input_shape = input_shape)(inputs)
    output = Dense(output_dim, activation='softmax')(x)

    # -------- Compile network model  --------------
    model = Model(inputs, output)

    optimizer=optimizers.SGD(lr=learning_rate, nesterov=nesterov, momentum=momentum)
    model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # print out network description
    # model.summary()

    history = model.fit(X_train, Y_train, validation_split=validation_split, epochs=epochs, batch_size=batch_size)

    return history 



def plot_effect_of_learning_rate(learning_rates=[1E-0, 1E-2, 1E-4], nesterov=False):

    plt.figure(figsize=(18, 15))
    # create learning curves for different learning rates
    for i in range(len(learning_rates)):
        # determine the plot number
        plot_no = 420 + (i+1)
        plt.subplot(plot_no)
        # fit model and plot learning curves for a learning rate
        history = mnist_fit_FCNet_model(learning_rate=learning_rates[i])
        # plot learning curves
        plt.plot(history.history['accuracy'], label='train')
        plt.plot(history.history['val_accuracy'], label='validation')
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title('Learning Rate='+str(learning_rates[i]), pad=-50)
    
    # save learning curves
    plt.savefig('plot_effect_of_learning_rate' + ".pdf")


def cifar10_fit_FCNet_model(learning_rate, batch_size, nesterov=False, momentum=0.0):

       # Load data and preprocess
    (X_train_CIFAR10, Y_train_CIFAR10), (X_test_CIFAR10,
                                         Y_test_CIFAR10) =  cifar10.load_data(PATH_CIFAR10)


    # dataset properties
    train_samples = X_train_CIFAR10.shape[0]
    test_samples = X_test_CIFAR10.shape[0]
    input_height = X_train_CIFAR10.shape[1]
    input_width = X_train_CIFAR10.shape[2]

    # three channels only for CIFAR10 as an input to the network (RGB)
    input_channels = 3


    # flatten images for input to a fully connected dense network
    X_train = X_train_CIFAR10.reshape(
        train_samples, input_height * input_width * input_channels)
    X_test = X_test_CIFAR10.reshape(
        test_samples, input_height * input_width * input_channels)

    # Convert training labels to categorical data (0 ... 1 .... 0), one-hot - to provide proper form to compute loss
    Y_train = to_categorical(Y_train_CIFAR10)
    # Test labels for trained model testing on the test set
    Y_test = Y_test_CIFAR10.flatten()

    # --- properties of network input and output format
    input_dims = X_train.shape[1]
    num_classes = Y_train.shape[1]
   
    # --- Preprocessing and normalizing input

    # --- compute training set stats

    max_value = X_train.max()
    min_value = X_train.min()

    mean_value = X_train.mean()
    std_value = X_train.std()

    # --- applying scaling factor for train and test
    rescaling_max = max_value

    # --- rescale input
    # Normalize between 0 and 1
    X_train = X_train.astype('float32') / rescaling_max
    # ! USE ON TEST only TRAIN normalizing factors !
    # Otherwise considered as sneaking information from train to test
    X_test = X_test.astype('float32') / rescaling_max

    # ---- zero mean, unit variance

    # mean = 0 ; standard deviation = 1.0

    mean_train = X_train.mean()
    std_train = X_train.std()
    X_train = (X_train - mean_train)/(std_train + 0.1)
    # -- transforming test using scaling parameters from X_train - trainings set !!!
    X_test = (X_test - mean_train)/(std_train + 0.1)
    
    SEED_NUM = 16823
    np.random.seed(SEED_NUM)  # for reproducibility

    # Shape of input to the network: spatial dimensions (here 1 dim flat vector), and channels number
    # (e.g 3 for RGB, here 1)
    input_shape = (input_dims, input_channels)

    hidden_dim = 256  # 128, 256, 512, ...
    output_dim = num_classes
    activation_func = 'relu'  # sigmoid, tanh, relu, elu, ...
    num_layers = 3

    # ---- learning rate:
    # sets the step size for each weight update executed by gradient descent
    # this will determine the stability and speed of convergence during learning
    # learning rate too low - very slow convergence, can stall learning entirely
    # learning rate too high - instable, may diverge so that learning collapses entirely
    learning_rate = learning_rate

    # ---- size of the mini batch
    # too small: computationally inefficient, no advantage through vectorization; slower convergence via high noise
    # too high: lack of noise, stochasticity in updates; danger of convergence to bad (flat) local minima
    batch_size = batch_size  # 32; 128; 256

    # Number of epochs: one epoch is going through the whole dataset
    epochs = 5

    # part of training dataset to be used for validation
    validation_split = 1/12  # 5000 out of 60000; 55000 train, 5000 validation

    # experiment with weight initialization only with deeper networks;
    # RELU the same - the deeper the network, the less stable training; additional measures neccessary

    # -------- Network Barebone Description, Functional Keras API --------------

    # --- different types of network weight initializers
    # initializer = tf.keras.initializers.GlorotUniform()
    # initializer = tf.keras.initializers.HeUniform()
    # initializer = tf.keras.initializers.LecunUniform()
    initializer = tf.keras.initializers.GlorotNormal()
    # initializer = tf.keras.initializers.HeNormal()
    # initializer = tf.keras.initializers.LecunNormal()
    # same as LecunNormal - recommended for selu in FC networks
    # initializer = tf.keras.initializers.VarianceScaling(scale=1., mode='fan_in', distribution='normal')

    # --- Building Network using Functional API ---

    # prepare input placeholder
    inputs = Input(shape=(input_dims, ))

    x = Dense(hidden_dim, activation=activation_func,
              kernel_initializer=initializer, input_shape=input_shape)(inputs)

    layer_count = 1
    for i in range(num_layers):
        print("Adding layer: ", layer_count)
        layer_count = layer_count + 1
        x = Dense(hidden_dim, activation=activation_func,
                  kernel_initializer=initializer)(x)

    output = Dense(output_dim, activation='softmax')(x)

    # -------- Compile network model  --------------
    model = Model(inputs, output)

    optimizer=optimizers.SGD(lr=learning_rate, nesterov=nesterov, momentum=momentum)
    model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # print out network description
    model.summary()

    # perform training
    history = model.fit(X_train, Y_train, validation_split=validation_split,
                        epochs=epochs, batch_size=batch_size)
    
    return history 


def cifar10_fit_ConvNet_model(learning_rate, batch_size, nesterov=False, momentum=0.0):
    
    # Load data and preprocess
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data(PATH_CIFAR10)

    # three channels only for CIFAR10 as an input to the network (RGB)
    input_channels = 3

    # --- properties of network input and output format
    input_dims = X_train.shape[1]
    num_classes = Y_train.shape[1]

    # --- Preprocessing and normalizing input

    # --- compute training set stats

    max_value = X_train.max()
    min_value = X_train.min()

    mean_value = X_train.mean()
    std_value = X_train.std()

    # --- applying scaling factor for train and test
    rescaling_max = max_value

    # --- rescale input
    # Normalize between 0 and 1
    X_train = X_train.astype('float32') / rescaling_max
    # ! USE ON TEST only TRAIN normalizing factors !
    # Otherwise considered as sneaking information from train to test
    X_test = X_test.astype('float32') / rescaling_max

    # ---- zero mean, unit variance

    # mean = 0 ; standard deviation = 1.0

    mean_train = X_train.mean()
    std_train = X_train.std()
    X_train = (X_train - mean_train)/(std_train + 0.1)
    # -- transforming test using scaling parameters from X_train - trainings set !!!
    X_test = (X_test - mean_train)/(std_train + 0.1)

    
    SEED_NUM = 16823
    np.random.seed(SEED_NUM)  # for reproducibility

    kernel_size = 3

    # number of feature/filter maps to be used in the layers
    # num_maps = [2]
    # num_maps = [2, 4]
    # num_maps = [4, 8, 16]
    # num_maps = [16, 32, 64]
    num_maps = [16, 32, 64, 128]
    num_layers = len(num_maps) - 1


    # --- properties of network input and output format
    # input_dims = 784
    num_classes = 10

    # Shape of input to the network: spatial dimensions (here 1 dim flat vector), and channels number
    # (e.g 3 for RGB, here 1)
    # input_shape = (input_dims, input_channels)
    validation_split = 1/12

    output_dim = num_classes
    activation_func = 'relu'  # sigmoid, tanh, relu, elu, ...

    # ---- learning rate:
    # sets the step size for each weight update executed by gradient descent
    # this will determine the stability and speed of convergence during learning
    # learning rate too low - very slow convergence, can stall learning entirely
    # learning rate too high - instable, may diverge so that learning collapses entirely
    learning_rate = learning_rate

    # ---- size of the mini batch
    # too small: computationally inefficient, no advantage through vectorization; slower convergence via high noise
    # too high: lack of noise, stochasticity in updates; danger of convergence to bad (flat) local minima
    batch_size = batch_size  # 32; 128; 256

    # Number of epochs: one epoch is going through the whole dataset
    epochs = 5

    # part of training dataset to be used for validation
    # validation_split = 1/12 # 5000 out of 60000; 55000 train, 5000 validation

    # experiment with weight initialization only with deeper networks;
    # The deeper the network, the less stable training without proper weight init

    # -------- Network Barebone Description, Functional Keras API --------------

    # --- Different weight initializers for the conv layers
    initializer = tf.keras.initializers.HeUniform()
    # initializer = tf.keras.initializers.GlorotNormal()
    # initializer = tf.keras.initializers.VarianceScaling(scale=1., mode='fan_in', distribution='normal')
    # initializer = tf.keras.initializers.LecunNormal() # e.g for using with SELU

    # --- Building Convolutional Network using Functional API ---

    # CIFAR10 input dimensions : width, height, channels
    # input_dim = (32, 32, 3) # CIFAR10: 32;
    # here we are not required to provide image dimensions,
    # as we work with fully convolutional network that does not use Flatten + Dense Layer operation
    # Size of the input image can be therefore variable, which is indicated with "None"
    input_dim = (None, None, 3)

    # three channels as an input to the network (RGB)
    input_channels = 3

    # -------- define Conv Net Model Architecture --------------
    # prepare input placeholder
    input_shape = input_dim
    inputs = Input(shape=input_dim)

   
    layer_count = 0
    x = Conv2D(filters=num_maps[layer_count], kernel_size=kernel_size, activation=activation_func,
               padding='same', kernel_initializer=initializer, input_shape=input_shape)(inputs)
    x = Conv2D(filters=num_maps[layer_count], kernel_size=kernel_size,
               activation=activation_func, padding='same', kernel_initializer=initializer)(x)

    layer_count = 1
    for i in range(num_layers):
        print("Adding block: ", layer_count)
        x = Conv2D(filters=num_maps[layer_count], kernel_size=kernel_size, strides=2,
                   activation=activation_func, padding='same', kernel_initializer=initializer)(x)

        x = Conv2D(filters=num_maps[layer_count], kernel_size=kernel_size,
                   activation=activation_func, padding='same', kernel_initializer=initializer)(x)

        layer_count = layer_count + 1

    x = Conv2D(filters=10, kernel_size=(1, 1), activation=activation_func,
               padding='same', kernel_initializer=initializer)(x)
    x = GlobalAveragePooling2D()(x)
    output = Activation('softmax')(x)

    # -------- Compile network model  --------------
    model = Model(inputs, output)

   
    optimizer=optimizers.SGD(lr=learning_rate, nesterov=nesterov, momentum=momentum)
    model.compile(optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # print out network description
    model.summary()


    # perform training
    history = model.fit(X_train, Y_train, validation_split=validation_split, epochs=epochs, batch_size=batch_size)
    
    return history 




def plot_effect_of_batch_sizes_loss(batch_sizes):

    fig, ax = plt.subplots(1, 2, figsize=(20, 5))
    for batch_size in batch_sizes:
        
        history = cifar10_fit_ConvNet_model(learning_rate=0.1, batch_size=batch_size).history
        ax[0].plot(history['loss'], label='batch_{}'.format(batch_size))
        ax[1].plot(history['val_loss'], label='batch_{}'.format(batch_size))
        
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Training loss')
    ax[0].set_title('Training loss for different batch sizes')
    ax[0].legend()

    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Validation loss')
    ax[1].set_title('Validation loss for different batch sizes')
    ax[1].legend()
    
    fig.savefig('plot_effect_of_batch_sizes_loss' + ".pdf")
    

def plot_effect_of_batch_sizes_acc(batch_sizes):

    fig, ax = plt.subplots(1, 2, figsize=(20, 5))
    for batch_size in batch_sizes:
        
        history = cifar10_fit_ConvNet_model(learning_rate=0.1, batch_size=batch_size).history
        ax[0].plot(history['accuracy'], label='batch_{}'.format(batch_size))
        ax[1].plot(history['val_accuracy'], label='batch_{}'.format(batch_size))
        
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Training accuracy')
    ax[0].set_title('Training accuracy for different batch sizes')
    ax[0].legend()

    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Validation accuracy')
    ax[1].set_title('Validation accuracy for different batch sizes')
    ax[1].legend()
    
    fig.savefig('plot_effect_of_batch_sizes_acc' + ".pdf")

    

if __name__ == "__main__":

#     (X_train_MNIST, Y_train_MNIST), (X_test_MNIST, Y_test_MNIST) = load_MNIST(PATH_MNIST)
#     # Task 1.1: Plot a selection of the input images using plot_data method above
#     plot_data(X_train_MNIST, Y_train_MNIST)
#     # Plot the label distribution 
#     plot_label_distribution(Y_train_MNIST)

#     # Task 1.2: Employ the use of `mnist_fit_FCNet_model` and `plot_accuracy` to plot the training/validation curves. 
#     history = mnist_fit_FCNet_model_on_unnormalised_data(learning_rate = 0.01, epochs=5,batch_size=32, nesterov=False)
#     plot_accuracy(history, 'plot_mnist_fit_FCNet_model_on_unnormalised_data')


#     # Task: train the FC-Net model on normalized dataset and plot accuracy
#     # train the FC-Net model on normalized dataset and plot accuracy
#     history = mnist_fit_FCNet_model(learning_rate=0.01)
#     plot_accuracy(history, 'plot_mnist_fit_FCNet_model')

#     # Task 2.1:  plug in your favorite learning rates
#     learning_rates = [1E-0, 1E-2, 1E-4]
#     plot_effect_of_learning_rate(learning_rates)

#     # Task 3.1
#     history = cifar10_fit_FCNet_model(learning_rate=0.1, batch_size=32)
#     plot_accuracy(history, 'plot_cifar10_fit_FCNet_model')


#     # Task 3.2: train the model using cifar10_fit_ConvNet_model with different learning rate and  plot the accuracy
#     history = cifar10_fit_ConvNet_model(learning_rate=0.1, batch_size=32)
#     plot_accuracy(history, 'plot_cifar10_fit_ConvNet_model')

#     history = cifar10_fit_ConvNet_model(learning_rate=0.01, batch_size=32)
#     plot_accuracy(history, 'plot_cifar10_fit_ConvNet_model_2')

#     # Task 3.3: train the model using cifar10_fit_ConvNet_model with nesterov=True, setting momentum=0.9, 0.95, . and plot the accuracy 
#     history = cifar10_fit_ConvNet_model(learning_rate=0.01, batch_size=32, nesterov=True, momentum=0.9)
#     plot_accuracy(history, 'plot_cifar10_fit_ConvNet_model_with_momentum')

    ### Task 4.1: Train the **ConvNet** model on different batch sizes ranging from **1 to 256**. Plot the training/validation curves using the `plot_effect_of_batch_sizes`.
    batch_sizes = [8, 64, 512]

    plot_effect_of_batch_sizes_loss(batch_sizes)
    plot_effect_of_batch_sizes_acc(batch_sizes)
    
