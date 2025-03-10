import numpy as np  # Import the numpy library for numerical operations
import keras  # Import the keras library for building neural networks
from keras.datasets import mnist  # Import the MNIST dataset from keras
from keras.models import Model  # Import the Model class from keras
from keras.layers import Dense, Input  # Import Dense and Input layers from keras
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten  # Import additional layers from keras
from keras import backend as k  # Import the backend module from keras

# Load the MNIST dataset, which contains handwritten digits
# The dataset is split into training and testing sets
# x_train and x_test contain the image data for training and testing respectively
# y_train and y_test contain the corresponding labels for the training and testing data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# x_train and x_test are numpy arrays of shape (num_samples, 28, 28)
# Each element in x_train and x_test is a 28x28 array representing a grayscale image
# y_train and y_test are numpy arrays of shape (num_samples,)
# Each element in y_train and y_test is an integer representing the class label (0-9)

# Set the dimensions of the input images
img_rows, img_cols = 28, 28

# Reshape the data based on the backend image data format
if k.image_data_format() == 'channels_first':
    # If the backend uses 'channels_first', reshape the data to (samples, channels, rows, cols)
    # Here, 'channels' refers to the number of color channels in the image
    # For grayscale images like MNIST, there is only 1 channel
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    inpx = (1, img_rows, img_cols)
else:
    # If the backend uses 'channels_last', reshape the data to (samples, rows, cols, channels)
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    inpx = (img_rows, img_cols, 1)

# Convert the data type to float32 and normalize the pixel values to the range [0, 1]
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Convert class vectors (integers) to binary class matrices (one-hot encoding)
# One-hot encoding is a process that converts class labels (integers) into a binary matrix representation
# For example, if there are 10 classes (digits 0-9), each class label is converted into a binary vector of length 10
# The vector has a 1 at the index corresponding to the class label and 0s elsewhere
# For instance, the class label 3 would be converted to [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
# This is useful for training neural networks, as it allows the model to output probabilities for each class
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

# Define the input layer with the shape of the input images
# The Input layer is the entry point of the neural network
# It specifies the shape of the input data that the network will receive
# 'shape=inpx' means the input data will have the shape defined by the variable 'inpx'
# 'inpx' is a tuple that represents the dimensions of the input images
# For example, if 'channels_first' is used, 'inpx' will be (1, 28, 28)
# This means the input data will have 1 channel and dimensions 28x28 (grayscale image)
# If 'channels_last' is used, 'inpx' will be (28, 28, 1)
# This means the input data will have dimensions 28x28 and 1 channel (grayscale image)
inpx = Input(shape=inpx)
# Define the first convolutional layer with 32 filters, kernel size of 3x3, and ReLU activation
# Conv2D is a 2D convolutional layer that applies a number of convolution filters to the input data
# '32' specifies the number of filters (also known as kernels) to be used in this layer
# Each filter will produce a feature map, so the output of this layer will have 32 feature maps
# 'kernel_size=(3, 3)' specifies the dimensions of the convolution filters (3x3 pixels)
# The filters will slide over the input image to detect patterns and features
# 'activation='relu'' specifies the activation function to be used
# ReLU (Rectified Linear Unit) is a common activation function that introduces non-linearity
# It outputs the input directly if it is positive, otherwise, it outputs zero
# 'inpx' is the input to this layer, which is the output from the previous layer (or the input layer)
layer1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(inpx)
# Define the second convolutional layer with 64 filters, kernel size of 3x3, and ReLU activation
layer2 = Conv2D(64, (3, 3), activation='relu')(layer1)
# Define the max pooling layer with a pool size of 3x3
# MaxPooling2D is a 2D pooling layer that performs max pooling operation
# Pooling layers are used to reduce the spatial dimensions (width and height) of the input volume
# This helps to reduce the number of parameters and computation in the network, and also controls overfitting
# 'pool_size=(3, 3)' specifies the size of the pooling window (3x3 pixels)
# The pooling window slides over the input feature map and takes the maximum value within the window
# This operation reduces the size of the feature map while retaining the most important features
# 'layer2' is the input to this layer, which is the output from the previous convolutional layer
layer3 = MaxPooling2D(pool_size=(3, 3))(layer2)
# Define the dropout layer with a dropout rate of 0.5 to prevent overfitting
# Dropout is a regularization technique used to prevent overfitting in neural networks
# Overfitting occurs when a model learns the training data too well, including its noise and outliers
# This results in poor generalization to new, unseen data, as the model becomes too specialized to the training data
# Dropout works by randomly setting a fraction of input units to 0 at each update during training time
# This prevents the model from relying too much on any individual neuron and forces it to learn more robust features
# '0.5' specifies the dropout rate, which is the fraction of input units to drop (set to 0)
# In this case, 50% of the input units will be dropped randomly during each training iteration
# 'layer3' is the input to this layer, which is the output from the previous max pooling layer
layer4 = Dropout(0.5)(layer3)
# Flatten the output from the previous layer to create a 1D feature vector
# Flatten is a layer that converts the multi-dimensional output of the previous layer into a 1D feature vector
# This is necessary before feeding the data into fully connected (dense) layers
# The previous layer (layer4) produces a 3D output (height, width, channels)
# Flatten takes this 3D output and reshapes it into a 1D array (vector)
# For example, if the output from the previous layer is of shape (batch_size, height, width, channels)
# 'batch_size' refers to the number of samples processed together in one forward/backward pass
# It is a hyperparameter that defines the number of training examples utilized in one iteration
# For instance, if batch_size is 500, the model processes 500 samples at a time before updating the weights
# Flatten will convert the output to shape (batch_size, height * width * channels)
# This allows the dense layers to process the data as a single vector of features
# Example: If the output from the previous layer is of shape (500, 5, 5, 32)
# Flatten will convert it to shape (500, 5 * 5 * 32) = (500, 800)
# 'layer4' is the input to this layer, which is the output from the previous dropout layer
layer5 = Flatten()(layer4)
# Define a dense (fully connected) layer with 250 units and sigmoid activation
# Dense is a fully connected layer where each neuron is connected to every neuron in the previous layer
# '250' specifies the number of neurons (units) in this layer
# Each neuron in this layer will receive input from all the neurons in the previous layer (layer5)
# Neurons are the basic units of a neural network that process input data and pass the result to the next layer
# Each neuron applies a linear transformation to the input data (weights and biases) and then applies an activation function
# 'activation='sigmoid'' specifies the activation function to be used
# Activation functions introduce non-linearity into the model, allowing it to learn complex patterns
# Without non-linearity, the model would be equivalent to a linear regression model, regardless of the number of layers
# Sigmoid activation function outputs a value between 0 and 1, which can be interpreted as a probability
# It is defined as sigmoid(x) = 1 / (1 + exp(-x))
# Example: If the input to a neuron is x = 0.5, the sigmoid activation function will output sigmoid(0.5) ≈ 0.62
# This activation function is useful for binary classification problems and for introducing non-linearity
# 'layer5' is the input to this layer, which is the output from the previous flatten layer
layer6 = Dense(250, activation='sigmoid')(layer5)
# Define the output layer with 10 units (one for each class) and softmax activation
# Each neuron in this layer corresponds to one of the 10 classes (digits 0-9) in the MNIST dataset
# 'activation='softmax'' specifies the activation function to be used
# Softmax activation function outputs a probability distribution over the classes
# It ensures that the sum of the output probabilities is equal to 1
# This is useful for multi-class classification problems where each input belongs to one of the multiple classes
# The softmax function is defined as softmax(x_i) = exp(x_i) / sum(exp(x_j)) for all j
# Example: If the input to the softmax layer is [2.0, 1.0, 0.1], the softmax output will be [0.659, 0.242, 0.099]
# This means the model assigns a probability of 65.9% to the first class, 24.2% to the second class, and 9.9% to the third class
# 'layer6' is the input to this layer, which is the output from the previous dense layer
layer7 = Dense(10, activation='softmax')(layer6)

# Create the model by specifying the input and output layers
# The Model class in Keras is used to create a model by specifying the input and output layers
# 'inpx' is the input layer of the model, which defines the shape of the input data
# 'layer7' is the output layer of the model, which defines the final output of the network
# The model is created by connecting the input layer to the output layer through the intermediate layers
# This defines the architecture of the neural network, specifying how data flows from the input to the output
# The model will take input data of the shape defined by 'inpx' and produce output data of the shape defined by 'layer7'
# In this case, the input data is a 28x28 grayscale image, and the output data is a probability distribution over 10 classes
model = Model([inpx], layer7)

# Compile the model with the Adadelta optimizer, categorical crossentropy loss, and accuracy metric
# Compiling the model configures the learning process before training

# Optimizers:
# Optimizers are algorithms used to update the weights of the neural network to minimize the loss function
# Common optimizers include:
# - SGD (Stochastic Gradient Descent): Updates weights using the gradient of the loss function with respect to the weights
# - Adam (Adaptive Moment Estimation): Combines the advantages of two other extensions of SGD, AdaGrad and RMSProp
# - RMSProp (Root Mean Square Propagation): Adapts the learning rate for each parameter
# - Adadelta: An extension of Adagrad that seeks to reduce its aggressive, monotonically decreasing learning rate

# 'optimizer=keras.optimizers.Adadelta()' specifies the optimizer to be used during training
# Adadelta is an adaptive learning rate optimization algorithm that adjusts the learning rate dynamically
# It is particularly useful for training deep neural networks with large datasets

# Loss Functions:
# Loss functions measure how well the model's predictions match the true labels
# Common loss functions include:
# - Mean Squared Error (MSE): Measures the average squared difference between predicted and actual values
# - Binary Crossentropy: Used for binary classification problems
# - Categorical Crossentropy: Used for multi-class classification problems where each input belongs to one of the multiple classes
# - Sparse Categorical Crossentropy: Similar to categorical crossentropy but used when labels are provided as integers

# 'loss=keras.losses.categorical_crossentropy' specifies the loss function to be used
# Categorical crossentropy calculates the difference between the predicted probability distribution and the true distribution (one-hot encoded labels)

# Metrics:
# Metrics are used to evaluate the model's performance
# Common metrics include:
# - Accuracy: Measures the percentage of correct predictions
# - Precision: Measures the accuracy of positive predictions
# - Recall: Measures the ability of the model to find all relevant cases within a dataset
# - F1 Score: Harmonic mean of precision and recall

# 'metrics=['accuracy']' specifies the metric to be used for evaluating the model's performance
# Accuracy is a common metric for classification problems, measuring the percentage of correct predictions

# The model will be trained to minimize the loss function and maximize the accuracy
model.compile(optimizer=keras.optimizers.Adadelta(),
              loss=keras.losses.categorical_crossentropy,
              metrics=['accuracy'])

# Train the model with the training data for 12 epochs and a batch size of 500
# 'epochs=12' specifies the number of epochs for training
# An epoch is one complete pass through the entire training dataset
# During each epoch, the model processes all the training samples once
# Multiple epochs are used to improve the model's performance by allowing it to learn from the data multiple times
# For example, if there are 60,000 training samples and the batch size is 500, the model will process 120 batches per epoch
# After 12 epochs, the model will have processed the entire training dataset 12 times
# 'batch_size=500' specifies the number of samples processed together in one forward/backward pass
# It is a hyperparameter that defines the number of training examples utilized in one iteration
# For instance, if batch_size is 500, the model processes 500 samples at a time before updating the weights
model.fit(x_train, y_train, epochs=12, batch_size=500)

# Evaluate the model with the test data and store the loss and accuracy
# 'model.evaluate' is used to evaluate the performance of the model on the test data
# 'x_test' is the input data for testing
# 'y_test' is the true labels for the test data
# 'verbose=0' means that the evaluation process will not produce any output
# The method returns the loss value and metrics values for the model
score = model.evaluate(x_test, y_test, verbose=0)

# Print the loss and accuracy
# 'score[0]' contains the loss value
# 'score[1]' contains the accuracy value
print('loss=', score[0])
print('accuracy=', score[1])



