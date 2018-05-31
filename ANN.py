from os import listdir
from os.path import isfile, join
from PIL import Image
import numpy as np
import math, random

W1 = None # Weight matrix for input to hidden
W3 = None # Weight matrix for hidden to output
bh1 = None
bh3 = None


def load(model_file):
    """
    Loads the network from the model_file
    :param model_file: file onto which the network is saved
    :return: the network
    """
    return pickle.load(open(model_file))

class NeuralNetwork(object):
    """
    Implementation of an Artificial Neural Network
    """
    def __init__(self, input_dim, hidden_size, output_dim,W3=None, learning_rate=0.01, reg_lambda=0.01):
        """
        Initialize the network with input, output sizes, weights and biases
        :param input_dim: input dim
        :param hidden_size: number of hidden units
        :param output_dim: output dim
        :param learning_rate: learning rate alpha
        :param reg_lambda: regularization rate lambda
        :return: None
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        if self.output_dim == 5:
        	# global W3
        	self.Wxh = W3
        else:
        	self.Wxh = np.random.randn(self.hidden_size, self.input_dim) * 0.01 # Weight matrix for input to hidden
        
        self.Why = np.random.randn(self.output_dim, self.hidden_size) * 0.01 # Weight matrix for hidden to output
        self.bh = np.zeros((self.hidden_size, 1)) # hidden bias
        self.by = np.zeros((self.output_dim, 1)) # output bias
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda

    def _feed_forward(self, X):
        """
        Performs forward pass of the ANN
        :param X: input
        :return: hidden activations, softmax probabilities for the output
        """
        # pdb.set_trace()
        # print self.Wxh
        h_a = np.tanh(np.dot(self.Wxh, np.reshape(X,(len(X),1))) + self.bh)
        if self.output_dim == 5:
        	ys = np.exp(np.dot(self.Why, h_a) + self.by)
	        probs = ys/np.sum(ys)
	        return h_a, probs
        else:        	
	        return h_a, np.dot(self.Why, h_a) + self.by

    def _regularize_weights(self, dWhy, dWxh, Why, Wxh):
        """
        Add regularization terms to the weights
        :param dWhy: weight derivative from hidden to output
        :param dWxh: weight derivative from input to hidden
        :param Why: weights from hidden to output
        :param Wxh: weights from input to hidden
        :return: dWhy, dWxh
        """
        dWhy += self.reg_lambda * Why
        dWxh += self.reg_lambda * Wxh
        return dWhy, dWxh

    def _update_parameter(self, dWxh, dbh, dWhy, dby):
        """
        Update the weights and biases during gradient descent
        :param dWxh: weight derivative from input to hidden
        :param dbh: bias derivative from input to hidden
        :param dWhy: weight derivative from hidden to output
        :param dby: bias derivative from hidden to output
        :return: None
        """
        self.Wxh += -self.learning_rate * dWxh
        self.bh += -self.learning_rate * dbh
        self.Why += -self.learning_rate * dWhy
        self.by += -self.learning_rate * dby

    def _back_propagation(self, X, t, h_a, probs):
        """
        Implementation of the backpropagation algorithm
        :param X: input
        :param t: target
        :param h_a: hidden activation from forward pass
        :param probs: softmax probabilities of output from forward pass
        :return: dWxh, dWhy, dbh, dby
        """
        dWxh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Why)
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by) 
        
        
        if self.output_dim == 5:
        	dy = np.copy(probs)
        	dy[t] -= 1
        else:     
        	dy = probs - t.reshape(self.output_dim,1)   	
        # print dy
        # dy[t] -= 1

        dWhy = np.dot(dy, h_a.T)
        dby += dy
        # pdb.set_trace()
        dh = np.dot(self.Why.T, dy)  # backprop into h
        dhraw = (1 - h_a * h_a) * dh # backprop through tanh nonlinearity
        dbh += dhraw
        # pdb.set_trace()
        dWxh += np.dot(dhraw, np.reshape(X, (len(X), 1)).T)
        return dWxh, dWhy, dbh, dby

    def _calc_smooth_loss(self, loss, len_examples, regularizer_type=None):
        """
        Calculate the smoothened loss over the set of examples
        :param loss: loss calculated for a sample
        :param len_examples: total number of samples in training + validation set
        :param regularizer_type: type of regularizer like L1, L2, Dropout
        :return: smooth loss
        """
        if regularizer_type == 'L2':
            # Add regulatization term to loss
            loss += self.reg_lambda/2 * (np.sum(np.square(self.Wxh)) + np.sum(np.square(self.Why)))
            return 1./len_examples * loss
        else:
            return 1./len_examples * loss

    def train(self, inputs, targets, validation_data, num_epochs, regularizer_type=None):
        """
        Trains the network by performing forward pass followed by backpropagation
        :param inputs: list of training inputs
        :param targets: list of corresponding training targets
        :param validation_data: tuple of (X,y) where X and y are inputs and targets
        :param num_epochs: number of epochs for training the model
        :param regularizer_type: type of regularizer like L1, L2, Dropout
        :return: None
        """
        for k in xrange(num_epochs):
            loss = 0
            for i in xrange(len(inputs)):
                # Forward pass
                # print inputs[i]
                h_a, probs = self._feed_forward(inputs[i])
                # loss += -np.log(probs[targets[i], 0])
                # loss += probs[targets[i], 0]

                # Backpropagation
                dWxh, dWhy, dbh, dby = self._back_propagation(inputs[i], targets[i], h_a, probs)

                # Perform the parameter update with gradient descent
                self._update_parameter(dWxh, dbh, dWhy, dby)

            # validation using the validation data

            # validation_inputs = validation_data[0]
            # validation_targets = validation_data[1]

            # print 'Validation'

            # for i in xrange(len(validation_inputs)):
            #     # Forward pass
            #     h_a, probs = self._feed_forward(inputs[i])
            #     loss += -np.log(probs[targets[i], 0])

            #     # Backpropagation
            #     dWxh, dWhy, dbh, dby = self._back_propagation(inputs[i], targets[i], h_a, probs)

            #     if regularizer_type == 'L2':
            #         dWhy, dWxh = self._regularize_weights(dWhy, dWxh, self.Why, self.Wxh)

            #     # Perform the parameter update with gradient descent
            #     self._update_parameter(dWxh, dbh, dWhy, dby)

            # if k%1 == 0:
            #     print "Epoch " + str(k) + " : Loss = " + str(self._calc_smooth_loss(loss, len(inputs), regularizer_type))
        
        if self.output_dim == 768:
        	global W1,bh1
        	W1 = self.Wxh
        	bh1 = self.bh
        elif self.output_dim == 16:
        	global W3,bh3
        	W3 = self.Wxh
        	bh3 = self.bh


    def predict(self, X):
        """
        Given an input X, emi
        :param X: test input
        :return: the output class
        """
        h_a, probs = self._feed_forward(X)
        # return probs
        # print X, probs
        return np.argmax(probs),h_a

    def save(self, model_file):
        """
        Saves the network to a file
        :param model_file: name of the file where the network should be saved
        :return: None
        """
        pickle.dump(self, open(model_file, 'wb'))

if __name__ == "__main__":
    mypath = '/home/deepak/Workspace/AML/see-lab/data'
    classes = [f for f in listdir(mypath)]    
    one_hot = {
    	'buddha':np.array([1,0,0,0,0]),
    	'butterfly':np.array([0,1,0,0,0]),
    	'airplanes':np.array([0,0,1,0,0]),
    	'Leopards':np.array([0,0,0,1,0]),
    	'Motorbikes':np.array([0,0,0,0,1])
    }
    
    data = []
    for _class in classes:
    	for instance in [f for f in listdir(mypath+'/'+_class)]:		
    		img = Image.open(mypath+'/'+_class+'/'+instance).convert('RGB')
    		rgb = np.array(img).reshape((1,768))
    		data.append([rgb[0], one_hot[_class]])
    data = np.asarray(data)
    np.random.shuffle(data)
    traindata = data[:int(math.floor(.8*len(data)))]
    testdata = data[int(math.floor(.8*len(data))):]

    nn1 = NeuralNetwork(768,16,768)
    nn2 = NeuralNetwork(16,8,16)
    
    # nn = NeuralNetwork(768,16,768)

    # for i in range(1000):
    #     num = random.randint(0,3)
    #     inp = np.zeros((4,))
    #     inp[num] = 1
    #     inputs.append(inp)
    #     targets.append(num)

    inputs = []
    targets = []
    inputs3 = []

    for rgb, onehot in traindata:
    	inputs.append(rgb)
    	inputs3.append(np.argmax(onehot))
        targets.append(rgb)

    nn1.train(inputs, targets, (inputs, targets), 2, regularizer_type='L2')

    inputs2 = []
    targets2 = []

    for inp in inputs:
    	x,prob=nn1.predict(inp)
    	inputs2.append(prob)

    nn2.train(inputs2, inputs2, (inputs2, inputs2), 2, regularizer_type='L2')
    nn3 = NeuralNetwork(16,8,5,W3)
    nn3.train(inputs2, inputs3, (inputs2, inputs3), 10, regularizer_type='L2')

    inputs = []
    targets = []
    a=0
    b = 0
    for rgb, onehot in testdata:
    	b+=1
    	x,prob=nn1.predict(rgb)    	
    	y,y2 = nn3.predict(prob)
    	if y == np.argmax(onehot): a+=1
    print('accuracy: ')
    print(a*100//b)