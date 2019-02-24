import numpy as np

from layers import *

class LogisticClassifier(object):
  """
  A logistic regression model with optional hidden layers.
  
  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """
  
  def __init__(self, input_dim=100, hidden_dim=None, weight_scale=1e-3, reg=0.0):
    """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {}
    self.reg = reg
    
    ############################################################################
    # TODO: Initialize the weights and biases of the model. Weights            #
    # should be initialized from a Gaussian with standard deviation equal to   #
    # weight_scale, and biases should be initialized to zero. All weights and  #
    # biases should be stored in the dictionary self.params, with first layer  #
    # weights and biases using the keys 'W1' and 'b1' and second layer weights #
    # and biases (if any) using the keys 'W2' and 'b2'.                        #
    ############################################################################
    if hidden_dim== None:
        W1 = np.random.normal(scale = weight_scale, size = (input_dim, 1))
        b1 = np.zeros(1)
        self.params['W1']=W1
        self.params['b1']=b1
    else:
        W1 = np.random.normal(scale = weight_scale, size = (input_dim, hidden_dim))
        b1 = np.zeros(hidden_dim)
        self.params['W1']=W1
        self.params['b1']=b1
        
        W2 = np.random.normal(scale = weight_scale, size = (hidden_dim, 1))
        b2 = np.zeros(1)
        self.params['W2']=W2
        self.params['b2']=b2
        
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################


  def loss(self, X, y=None):
    """
    Compute loss and gradient for a minibatch of data.

    Inputs:
    - X: Array of input data of shape (N, D)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N,) where scores[i] represents the logit for X[i]
    
    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """  
    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the model, computing the            #
    # scores for X and storing them in the scores variable.                    #
    ############################################################################
    hidden=False
    #print(X.shape)
    W1 = self.params['W1']
    #print(W1.shape)
    b1= self.params['b1']
    #print(b1.shape)
    fc_out, fc1_cache = fc_forward(X, W1, b1)
    #print(fc_out.shape)
    if fc_out.shape[1] != 1:
        hidden = True
        #has hidden
        W2=self.params['W2']
        b2=self.params['b2']
        fc_out,relu_cache = relu_forward(fc_out)
        fc_out, fc2_cache = fc_forward(fc_out, W2, b2)
    #print(hidden)
    #print('\n\n', fc_out)
    scores = fc_out.reshape(-1)
    #print(scores)
    '''
    N=X.shape[0]
    for i in range(N):
        x_i = X[i,:] # D
        h_i = np.dot(self.params['W1'], x_i) # h
        .................relu(h_i + self.params['b1'])
    '''
        
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If y is None then we are in test mode so just return scores
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the model. Store the loss          #
    # in the loss variable and gradients in the grads dictionary. Compute data #
    # loss and make sure that grads[k] holds the gradients for self.params[k]. #
    # Don't forget to add L2 regularization.                                   #
    #                                                                          #
    ############################################################################
    loss, dx = logistic_loss(scores, y)
    loss += 0.5 * self.reg * (np.linalg.norm(W1)**2)
    #print(dx.shape)
    dx = dx.reshape(dx.shape[0],-1)
    #print(dx.shape)
    if hidden:
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

import pickle
from solver import Solver
with open('data.pkl', 'rb') as f:
    d = pickle.load(f, encoding='latin1')
X=d[0]
y=d[1]
#y = 2*y - np.ones(y.shape[0])
data = {
        'X_train': X[:500],
        'y_train': y[:500],
        'X_val': X[500:750],
        'y_val': y[500:750]
}

model = LogisticClassifier(input_dim=20, hidden_dim=100, weight_scale=1e-1, reg=0.01)
solver = Solver(model, data, update_rule='sgd', optim_config={'learning_rate': 0.5,},
                lr_decay=0.95, num_epochs=100, batch_size=10, print_every=500)
solver.train()
x_test=X[750:]
y_test=y[750:]
print(solver.check_accuracy(x_test, y_test, batch_size=100))
