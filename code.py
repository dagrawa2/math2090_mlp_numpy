# Devanshu Agrawal
# MATH 2090
# May 2, 2014

# The Effect of Hidden Neurons on a Three-Layer Perceptron Classifier and Autoencoder

# We consider a multi-layer perceptron and train it using back propagation and stochastic gradient descent.
# We use the trained perceptron to classify 8-by-8 pixel images of handwritten 0's and 1's (i.e., the digits "0" and "1"),
# and we also use it to autoencode and reconstruct the input image.
# Thus, the perceptron has 64 input neurons, 65 output neurons (64 for autoencoding and 1 for classification), and h hidden neurons.
# We measure the error in classification and autoencoding as a function of h.

# First, we define a class for neural networks.
# Note: This part of the code is a modification of the code found at
# http://rolisz.ro/2013/04/18/neural-networks-in-python/
# But all comments throughout the code are mine.

# Import floating point division and numpy

from __future__ import division
import numpy as np

# We have the option of two different activation functions-- the tanh function and the logistic function.
# For our purposes, we want all outputs in the interval [0, 1].
# So we transform tanh to lie in this interval, but we still call the function "tanh."
# For both activation functions, we also define their derivatives in terms of the output.

def tanh(x):
	return ( np.tanh(x) + 1)/2.0

def tanh_deriv(y):
	return ( 1.0 - (2*y-1)**2 )/2.0

def logistic(x):
	return 1/(1 + np.exp(-x))

def logistic_derivative(y):
	return y*(1.0 - y)

# Define the NeuralNetwork class.

class NeuralNetwork:

	def __init__(self, layers, activation='logistic'):
		"""
		:param layers: A list containing the number of units in each layer. Should be at least two values.
		:param activation: The activation function to be used. Can be "logistic" or "tanh."
		"""
		if activation == 'logistic':
			self.activation = logistic
			self.activation_deriv = logistic_derivative
		elif activation == 'tanh':
			self.activation = tanh
			self.activation_deriv = tanh_deriv

# The activation function is an attribute of the NeuralNetwork "self."
# Define the "weights" and "bias" attributes.
# self.weights is a list of weight arrays whose entries are initialized randomly between -0.25 and 0.25.
# self.bias is a list of arrays, each with one row, whose entries are initialized to 0.

		self.weights = []
		self.bias = []
		for i in range(0, len(layers) - 1):
			self.weights.append( (2*np.random.random((layers[i], layers[i+1]))-1)*0.25 )
			self.bias.append( np.zeros([1, layers[i+1]]) )

# Define the fit method, which uses back propagation and stochastic gradient descent to train "self."

	def fit(self, X, y, learning_rate=0.2, epochs=10000):
		"""
		:param X: The training set of input vectors. Each row of X is an input.
		:param y: The training set of output values, or targets, corresponding to the rows of X.
		:param learning_rate: The learning rate of "self."
		:param epochs: The number of times stochastic gradient descent and back propagation are repeated for training.
		"""
		X = np.atleast_2d(X)
		y = np.array(y)

# In the kth epoch, crandomly choose an input from X and store it as a.
# Note that a is a list containing one array X[i].

		for k in range(epochs):
			i = np.random.randint(X.shape[0])
			a = [X[i]]

# Run the network and compute the activation energies at each layer of the network.
# Store the activations at each layer as arrays listed in a.
# Note the clever use of "append," which helps to reduce calling by indeces.
# Also note that self.bias[l] must be a 2D array with one row.

			for l in range(len(self.weights)):
				a.append(self.activation(np.dot(a[l], self.weights[l]) + self.bias[l]))

# Now use back propagation.
# By default, the squared error between the target output y[i[ and final activation output a[-1] is used:
# 	E(weights) = 1/2 * (y[i] - a[-1])^2.
# Calculate the deltas at each layer.
# Note that deltas is a list with one array.
# Note: To use cross entropy instead of squared error, simply let deltas = [error].
# If using cross entropy, make sure to use 'logistic' instead of 'tanh'.

			error = y[i] - a[-1]
			deltas = [error * self.activation_deriv(a[-1])]
			for l in range(len(a) - 2, 0, -1): # we need to begin at the second to last layer
				deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_deriv(a[l]))
			deltas.reverse()

# Update the weights and biases.

			for i in range(len(self.weights)):
				layer = np.atleast_2d(a[i])
				delta = np.atleast_2d(deltas[i])
				self.weights[i] += learning_rate * layer.T.dot(delta)
				self.bias[i] += learning_rate * delta

# We are done with the training method.
# Now define the prediction method, where "self.predict" takes a single input and returns an output.
# For This, there is no need to store all the activations-- just the final output.
# Since self.bias[l] is 2D, we require a to be 2D as well.
# Thus, we have the procedure return a[0], so that only the single row in a is returned as a 1D object.

	def predict(self, x):
		a = np.atleast_2d(x)
		for l in range(0, len(self.weights)):
			a = self.activation(np.dot(a, self.weights[l]) + self.bias[l])
		return a[0]

# We are done with the NeuralNetwork class.


# Now let us use a perceptron to classify and autoencode images of the digits "0" and "1."
# First we import some stuff from the `sklearn' python module,
# and then we load the digits data set.
# X is a 1797-by-64 array,
# where each row is an input vector representing the image of a digit.
# Y is a 1797-by-1 array listing the classes of the corresponding input vectors in X.
# We normalize X such that the lowest pixel intensity in the set is 0 and the greatest is 1.

from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score

digits = load_digits()
X = digits.data
y = digits.target
X -= X.min()
X /= X.max()

# X contains the images of all 10 digits, but we only want to work with zeros and ones.
# Z_in is a 360-by-64 array that is a subset of X-- only zeros and ones.
# Since we are autoencoding in addition to classifying, the corresponding output vectors must contain a copy of the input (for autoencoding) as well as the class.
# Z_out is a 360-by-65 array, where the first 64 entries of each row refer to the copy of the corresponding input vector in Z_in, and the 65th entry in each row is the class of that input.

Z_out = np.hstack(( X[y <= 1], np.atleast_2d(y[y <= 1]).T ))
Z_in = Z_out[:, :64]

# We use `train_test_split' to split the data set into a training set (X_train, y_train) and a testing set (X_test, y_test).
# We make sure that the first image of a zero and the first image of a one are not in the training set but instead in the testing set.
# (Note the first row of Z_in is an image of a zero, and the second is of a one.)
# We do this so that we can use the same testing images to visualize the reconstruction, regardless of the trial or experiment.

X_train, X_test, y_train, y_test = train_test_split(Z_in[2:,:], Z_out[2:,:])
X_test = np.vstack(( Z_in[:2,:], X_test ))
y_test = np.vstack(( Z_out[:2,:], y_test ))

# We have the input and output data sets ready, but let's prepare for images before running the project itself.
# Define a procedure `image' that converts a 64-vector into an 8-by-8 array.

def image(A):
	I = np.zeros([8, 8])
	for i in range(8):
		for j in range(8):
			I[i,j] = A[8*i+j]
	return I

# Let us go ahead and store the first test images for zero and one.

image_zero_in = image(y_test[0,:64])
image_one_in = image(y_test[1,:64])

# Finally, we are ready to run the project proper.
# First we define a list H of all the numbers of hidden neurons for which we would like to test our perceptron.
# For each h in H, we define a perceptron `NN' with 64 input neurons, h hidden neurons, and 65 output neurons.
# We train `NN' using (X_train, y_train).
# Then we use the trained `NN' to predict the value for each input listed in X_test, and we store the predictions in the array 'out'.
# For each h, we compute the area under the "receiver operating characteristic" (ROC) curve to measure the accuracy of classification, and we compute the root mean square (RMS) error between the input and reconstructed output to measure the accuracy of autoencoding.
# These computed values are appended to the list roc and error_rms respectively.
# Additionally, for h = 1 and h = 32, we store the reconstructed outputs of the first two test inputs as images to be plotted later.

H = [1,2,4,8,16,32]

roc = []
error_rms = []
for h in H:
	nn = NeuralNetwork([64,h,65],'tanh')
	nn.fit(X_train,y_train,epochs=5000)
	out_list = []
	for i in range(X_test.shape[0]):
		out_list.append( nn.predict( X_test[i] ) )
		out = np.array(out_list)
	roc.append( roc_auc_score(y_test[:,64], out[:,64]) )
	error = np.sqrt( np.sum( (y_test[:,:64]-out[:,:64])**2 )/np.size(y_test[:,:64]) )
	error_rms.append( error )
	if (h == 1):
		image_zero_out_1 = image(out[0,:64])
		image_one_out_1 = image(out[1,:64])
	if (h == 32):
		image_zero_out_32 = image(out[0,:64])
		image_one_out_32 = image(out[1,:64])

# We finish the project by generating two figures.
# The figure `fig1' is a plot of the autoencoding RMS error against the number of hidden neurons used.
# The figure 'fig2' gives examples of autoencoding by plotting the reconstructed outputs for the first two inputs in X_test (a zero and a one). For each input, two reconstructions are plotted-- one using h = 1 and another using h = 32.
# By default, the plots are displayed as standard output (in a GUI).
# To save the plots to files, comment out 'plt.show()' and uncomment the fig.savefig(. . .) lines

import matplotlib.pyplot as plt

fig1 = plt.figure()
sub = fig1.add_subplot(1,1,1)
sub.plot(H, error_rms)
sub.set_title('Error in Autoencoding')
sub.set_xlabel('Number of Hidden Neurons')
sub.set_ylabel('RMS Error')
#fig1.savefig('NeuralError.png', bbox_inches='tight')
plt.show()

plt.set_cmap("gray")
fig2 = plt.figure()
sub1 = fig2.add_subplot(2,3,1)
sub1.imshow(image_zero_in)
sub1.set_title('Inputs')
sub2 = fig2.add_subplot(2,3,2)
sub2.imshow(image_zero_out_1)
sub2.set_title('Outputs (h = 1)')
sub3 = fig2.add_subplot(2,3,3)
sub3.imshow(image_zero_out_32)
sub3.set_title('Outputs (h = 32)')
sub4 = fig2.add_subplot(2,3,4)
sub4.imshow(image_one_in)
sub5 = fig2.add_subplot(2,3,5)
sub5.imshow(image_one_out_1)
sub6 = fig2.add_subplot(2,3,6)
sub6.imshow(image_one_out_32)
#fig2.savefig('NeuralImages.png', bbox_inches='tight')
plt.show()