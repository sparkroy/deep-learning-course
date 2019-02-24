import numpy as np
from layers import *


class SoftmaxClassifier(object):
    """
    A fully-connected neural network with
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecture should be fc - softmax if no hidden layer.
    The architecture should be fc - relu - fc - softmax if one hidden layer

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=28*28, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg
        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with fc weights                                  #
        # and biases using the keys 'W' and 'b'                                    #
        ############################################################################
        if hidden_dim== None:
            W1 = np.random.normal(scale = weight_scale, size = (input_dim, num_classes))
            b1 = np.zeros(num_classes)
            self.params['W1']=W1
            self.params['b1']=b1
            self.hidden = False
        else:
            self.hidden = True
            W1 = np.random.normal(scale = weight_scale, size = (input_dim, hidden_dim))
            b1 = np.zeros(hidden_dim)
            self.params['W1']=W1
            self.params['b1']=b1
            
            W2 = np.random.normal(scale = weight_scale, size = (hidden_dim, num_classes))
            b2 = np.zeros(num_classes)
            self.params['W2']=W2
            self.params['b2']=b2
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the one-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        #print(X.shape)
        W1 = self.params['W1']
        #print(W1.shape)
        b1= self.params['b1']
        #print(b1.shape)
        fc_out, fc1_cache = fc_forward(X, W1, b1)
        
        if self.hidden:
            #print('has hidden!')
            #has hidden
            W2=self.params['W2']
            b2=self.params['b2']
            fc_out,relu_cache = relu_forward(fc_out)
            fc_out, fc2_cache = fc_forward(fc_out, W2, b2)
        #print(hidden)
        scores = fc_out
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the one-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, dx = softmax_loss(scores, y)
        loss += 0.5 * self.reg * (np.linalg.norm(W1)**2)
        if self.hidden:
            loss += 0.5 * self.reg * (np.linalg.norm(W2)**2)
            dx, dW2, db2 = fc_backward(dx, fc2_cache)
            grads['b2'] = db2
            grads['W2'] = dW2 + self.reg * W2
            dx = relu_backward(dx, relu_cache)
        #print(dx.shape, fc1_cache[0].shape,fc1_cache[1].shape,fc1_cache[2].shape)
        dx, dW1, db1 = fc_backward(dx, fc1_cache)
        grads['b1'] = db1
        grads['W1'] = dW1 + self.reg * W1
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

from solver import Solver
import tensorflow as tf
(x_train,y_train),(x_test, y_test)=tf.keras.datasets.mnist.load_data()
x_train=x_train.reshape(x_train.shape[0],-1)
x_test=x_test.reshape(x_test.shape[0],-1)
print(x_train.shape,x_test.shape)

#use small
#x_train=x_train[:10]
#y_train=y_train[:10]
###

val_size = int(x_train.shape[0]*0.1)
x_val = x_train[:val_size]
y_val = y_train[:val_size]
x_train = x_train[val_size:]
y_train = y_train[val_size:]


data = {
        'X_train': x_train,
        'y_train': y_train,
        'X_val': x_val,
        'y_val': y_val
}
print("model start!")
'''
model = SoftmaxClassifier(input_dim=28*28, hidden_dim=None, num_classes=10,
                 weight_scale=1e-3, reg=0.0)
solver = Solver(model, data, update_rule='sgd', optim_config={'learning_rate': 1e-5,},
                lr_decay=0.95, num_epochs=5, batch_size=100, print_every=100)
'''
model = SoftmaxClassifier(input_dim=28*28, hidden_dim=14*14, num_classes=10,
                 weight_scale=1e-3, reg=0.01)
solver = Solver(model, data, update_rule='sgd', optim_config={'learning_rate': 1e-3,},
                lr_decay=0.95, num_epochs=5, batch_size=100, print_every=100)

solver.train()
print(solver.check_accuracy(x_test, y_test, num_samples=None, batch_size=100))
