import numpy as np

from layers import *


class ConvNet(object):
  """
  A convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - fc - softmax

  You may also consider adding dropout layer or batch normalization layer. 
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(1, 28, 28), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    
    ##just implementation of one fully-connected layer (as shown in the architecture) is fine
    
    #b is # of neurons in conv layer
    #W is N * volume kernel
    C, H, W = input_dim
    
    W1 = np.random.normal(scale = weight_scale, size = (num_filters, C, filter_size, filter_size))
    #b1 = np.zeros(num_filters)
    
    HH=filter_size
    WW=filter_size
    H_ = H - HH + 1
    W_ = W - WW + 1
    '''
    W2 = np.random.normal(scale = weight_scale, size = (int(num_filters * H_/2 * W_/2), num_classes))
    b2 = np.zeros(num_classes)
    '''
    W2 = np.random.normal(scale = weight_scale, size = (int(num_filters * H_/2 * W_/2), hidden_dim))
    b2 = np.zeros(hidden_dim)
    
    W3 = np.random.normal(scale = weight_scale, size = (hidden_dim, num_classes))
    b3 = np.zeros(num_classes)
    
    #W3 and b3 are not used
    self.params['W1']=W1
    self.params['W2']=W2
    self.params['W3']=W3

    #self.params['b1']=b1
    self.params['b2']=b2
    self.params['b3']=b3
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.items():
    #for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)

  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    #W1, b1 = self.params['W1'], self.params['b1']
    W1 = self.params['W1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    
    ##conv - relu - 2x2 max pool - fc - softmax
    
    
    if len(X.shape)==3:
        X=X.reshape(X.shape[0],1,X.shape[1],X.shape[2])
    #print(X.shape)
    # no bias
    conv_out, conv_cache = conv_forward(X, W1)
    #print('conv out',conv_out.shape)
    #out: N, F, H', W'
    
    #relu
    relu_out, relu_cache = relu_forward(conv_out)
    #print('relu out',relu_out.shape)
    #maxpool
    pool_out, pool_cache = max_pool_forward(relu_out, pool_param)
    #print('pool out',pool_out.shape)
    #fc
    shape_before_fc = pool_out.shape
    pool_out = pool_out.reshape(pool_out.shape[0],-1)
    fc_out, fc_cache = fc_forward(pool_out, W2, b2)
    
    relu_out2, relu_cache2 = relu_forward(fc_out)
    fc_out2, fc_cache2 = fc_forward(relu_out2, W3, b3)
    
    #print('fc out',fc_out.shape)
    scores = fc_out2
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    
    #softmax_loss(x, y), x: score
    loss, dx = softmax_loss(scores, y)
    #print('loss dx',dx.shape)
    #reg + loss
    loss += 0.5 * self.reg * (np.linalg.norm(W1)**2 + np.linalg.norm(W2)**2 + np.linalg.norm(W3)**2)
    #fc
    dx, dW3, db3 = fc_backward(dx, fc_cache2)
    dW3 = self.reg * W3 + dW3
    #relu
    dx = relu_backward(dx, relu_cache2)
    #fc
    dx, dW2, db2 = fc_backward(dx, fc_cache)
    dW2 = self.reg * W2 + dW2
    #print('fc dx', dx.shape)

    #pool
    dx = dx.reshape(shape_before_fc)
    #print('fc dx, after reshape', dx.shape)
    dx = max_pool_backward(dx, pool_cache)
    #print('pool dx', dx.shape)
    #relu
    dx = relu_backward(dx, relu_cache)
    #print(dx.shape)
    #conv
    dx, dW1 = conv_backward(dx, conv_cache)
    #print(dx.shape)
    dW1 = self.reg * W1 + dW1
    #b1 is not used in conv.
    grads['b2'] = db2
    grads['W1'] = dW1
    grads['W2'] = dW2
    grads['b3'] = db3
    grads['W3'] = dW3
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads

'''
import torch
import torchvision
import torchvision.datasets as datasets
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)
'''
from solver import Solver
import tensorflow as tf
(x_train,y_train),(x_test, y_test)=tf.keras.datasets.mnist.load_data()

#use small
x_train=x_train[:10]
y_train=y_train[:10]
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
model = ConvNet(input_dim=(1, 28, 28), num_filters=32, filter_size=5,
               hidden_dim=50, num_classes=10, weight_scale=1e-3, reg=0.0)
solver = Solver(model, data, update_rule='sgd', optim_config={'learning_rate': 1e-2,},
                lr_decay=0.95, num_epochs=5, batch_size=100, print_every=100)
solver.train()
print(solver.check_accuracy(x_test, y_test, num_samples=None, batch_size=100))

