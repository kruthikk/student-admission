import numpy as np

#sigmoid function - a sigmoid maps any value, to a value between 0 and 1 .(used to convert a number to a.probability)
#derive = Tue, gives slope of the sigmoid function
def nonlinear(x, derive=False):
	if (derive==True):
		return x * (1-x)
	return 1/(1+np.exp(-x))

#input dataset - we have training examples, with each example having 3 nodes 
"""
 Each row is a single "training example". Each column corresponds to one of our input nodes. 
 Thus, we have 3 input nodes to the network and 4 training examples.
"""
X = np.array([ [0,0,1],
			[0,1,1],
			[1,0,1],
			[1,1,1]
	])
#our network has 3 inputs and 1 output for each example.
# This is the output I generated the dataset horizontally (with a single row and 4 columns) for space.
# ".T" is the transpose function. After the transpose, this y matrix has 4 rows with one column. 

#output dataset
"""
y = np.array([
	[0],
	[0],
	[1],
	[1]
	])
"""
y = np.array([[0,0,1,1]]).T

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

#initialise hidden layer syn0 - weights layer that is used to predict next input value
# initialize weights randomly with mean 0
# input = 4*3 and output 4 * 1, so we need hidden weight 3 *1
# or since 3 input nodes and 1 output node we need 3 *1 matrix
"""
This is our weight matrix for this neural network. It's called "syn0" to imply "synapse zero". 
Since we only have 2 layers (input and output), we only need one matrix of weights to connect them. 
Its dimension is (3,1) because we have 3 inputs and 1 output. Another way of looking at it is that l0 is 
of size 3 and l1 is of size 1. Thus, we want to connect every node in l0 to every node in l1,
 which requires a matrix of dimensionality (3,1)

All of the learning is stored in the syn0 matrix
"""
sync0 = 2 * np.random.random((3,1)) - 1

# start the learning process - This for loop "iterates" multiple times over the 
#training data  to optimize our network to the dataset.
print('input', X)
#print ('transpose of input', X.T)
print('output', y)
print('sync0', sync0)

epochs=10000
for iter in range(epochs):
	#forward propagation
	l0 = X
	#predict the output based on given input layer and take sigmoid
	# np.dot of (4*3).(3*1) = 4*1
	# 4 samples we get 4 guesses
	#l1 is a guess of output for each input
	l1 = nonlinear(np.dot(l0, sync0))

	#by how much did we miss = 4*1
	l1_error = y - l1
	#derivative of l1 - slopes of the sigmoid
	derivative_l1 = nonlinear(l1, True) # 4*1

	#calculate the slope of the sigmoid - reduce the error
	#The Error Weighted Derivative - multiple error to the slopes - derivative of l1
	l1_delta = l1_error * derivative_l1  # (4*1) * 4*1 = 4*1

	#print('l1', l1)
	#print('l1delta', l1_delta)
	#print('l0.T', l0.T)
	# now update the weights
	sync0 += np.dot(l0.T, l1_delta)
	#print('updatedsync0',sync0)

print('output of training')
print( l1)



