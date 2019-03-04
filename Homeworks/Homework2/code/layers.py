from builtins import range
import numpy as np
import math

def fc_forward(x, w, b):
    """
    Computes the forward pass for a fully-connected layer.

    The input x has shape (N, Din) and contains a minibatch of N
    examples, where each example x[i] has shape (Din,).

    Inputs:
    - x: A numpy array containing input data, of shape (N, Din)
    - w: A numpy array of weights, of shape (Din, Dout)
    - b: A numpy array of biases, of shape (Dout,)

    Returns a tuple of:
    - out: output, of shape (N, Dout)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the forward pass. Store the result in out.              #
    ###########################################################################
    N=x.shape[0]
    Dout = b.shape[0]
    out = np.zeros((N, Dout))
    
    '''
    wT=np.transpose(w)
    
    for i in range(N):
        x_i = x[i] # Din
        out[i,:] = np.dot(wT, x_i) + b
    '''
    out = np.dot(x, w) + b
        
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def fc_backward(dout, cache):
    """
    Computes the backward pass for a fully_connected layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, Dout)
    - cache: Tuple of:
      - x: Input data, of shape (N, Din)
      - w: Weights, of shape (Din, Dout)
      - b: Biases, of shape (Dout,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, Din)
    - dw: Gradient with respect to w, of shape (Din, Dout)
    - db: Gradient with respect to b, of shape (Dout,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    
    #dout is dL/dY
    dx = np.dot(dout, np.transpose(w))
    dw = np.dot(np.transpose(x), dout)
    
    db = np.sum(np.transpose(dout),1)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    out = x * (x>0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x

    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    dx = dout * (x > 0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        # 
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        miu = np.mean(x,axis=0)
        var = np.var(x, axis=0)
        running_mean = momentum * running_mean + (1 - momentum) * miu
        running_var = momentum * running_var + (1 - momentum) * var
        
        x_hat = (x - miu)/((var + eps)**0.5)
        out = gamma * x_hat + beta
        cache = (x, eps, gamma, miu, var, x_hat)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        miu = running_mean
        var = running_var
        x_hat = (x - miu)/((var + eps)**0.5)
        out = gamma * x_hat + beta
        cache = (x, eps, gamma, miu, var, x_hat)
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    x, eps, gamma, miu, var, x_hat = cache
    m = dout.shape[0]
    dbeta = np.sum(dout,axis=0)
    dgamma = np.sum(dout * x_hat, axis=0)
    
    dxhat = dout * gamma
    dvar = np.sum(dxhat * (x - miu) * (-0.5*(var + eps)**-1.5), axis=0)
    dmiu = np.sum(-dxhat / ((eps + var)**0.5), axis=0) + dvar * np.sum(-2*(x - miu),axis=0)/m
    dx = dxhat / ((var + eps)**0.5) + dvar * 2*(x - miu)/m + dmiu/m
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Implement the vanilla version of dropout.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        mask = np.random.uniform(size=x.shape)
        mask = mask < p
        out = x * mask
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        out = x
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        dx = dout * mask
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward(x, w):
    """
    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = H - HH + 1
      W' = W - WW + 1
    - cache: (x, w)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    #w = np.flip(np.flip(w,3),2)
    F, C, HH, WW = w.shape
    N, C, H, W = x.shape
    #stride = 1, pad = 0
    H_ = H - HH + 1
    W_ = W - WW + 1
    out = np.zeros((N, F, H_, W_))
    for i in range(H_):
        for j in range(W_):
            mask = x[:,:, i:i+HH, j:j+WW]
            for k in range(F):
                out[:,k,i,j]=np.sum(mask*w[k,:,:,:],axis=(1,2,3))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w)
    return out, cache


def conv_backward(dout, cache):
    """
    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w) as in conv_forward

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    """
    dx, dw = None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    x, w = cache
  
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    H_out = 1 + (H  - HH) 
    W_out = 1 + (W  - WW) 
    
    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
  
  
    for i in range(H_out):
        for j in range(W_out):
            mask = x[:, :, i:i+HH, j:j+WW]
            for k in range(F): #compute dw
                #dw[k ,: ,: ,:] += np.sum(mask * (dout[:, k, i, j])[:, None, None, None], axis=0)
                dw[k ,: ,: ,:] = np.add( dw[k ,: ,: ,:], np.sum(mask * (dout[:, k, i, j])[:, None, None, None], axis=0), out = dw[k ,: ,: ,:], casting='unsafe')
            for n in range(N): #compute dx
                #dx[n, :, i:i+HH, j:j+WW] += np.sum((w[:, :, :, :] * 
                                                 #(dout[n, :, i, j])[:,None ,None, None]), axis=0)
                dx[n, :, i:i+HH, j:j+WW] = np.add(dx[n, :, i:i+HH, j:j+WW], np.sum((w[:, :, :, :] * 
                                                 (dout[n, :, i, j])[:,None ,None, None]), axis=0), out = dx[n, :, i:i+HH, j:j+WW], casting = 'unsafe')
    #dw = np.flip(np.flip(dw,3),2)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw


def max_pool_forward(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    N, C, H, W = x.shape
    pool_h = pool_param['pool_height']
    pool_w = pool_param['pool_width']
    stride = pool_param['stride']
    H_ = int(1 + (H - pool_h) / stride)
    W_ = int(1 + (W - pool_w) / stride)
    
    out = np.zeros((N, C, H_, W_))
    
    
    for hh, h in enumerate(range(0, H, stride)):
        #print(hh)
        for ww, w in enumerate(range(0, W, stride)):
            #print(ww)
            if h+pool_h <= H and w+pool_w <= W:
                out[:, :, hh, ww] = np.amax(x[:, :, h:h+pool_h, w:w+pool_w].reshape(N,C,-1),axis=2)
            

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    x, pool_param = cache
    N, C, H, W = x.shape
    pool_h = pool_param['pool_height']
    pool_w = pool_param['pool_width']
    stride = pool_param['stride']
    dx = np.zeros(x.shape)
    _, __, H_, W_ = dout.shape
    '''
    for hh, h in enumerate(range(0, H, stride)):
        #print(hh)
        for ww, w in enumerate(range(0, W, stride)):
            #print(ww)
            max_id = np.argmax(x[:, :, h:h+pool_h, w:w+pool_w].reshape(N,C,-1),axis=2)
            max_r = max_id // pool_w + h
            max_c = np.remainder(max_id, pool_w) + w
            
            for n in range(N):
                for c in range(C):
                    dx[n,c,max_r[n,c], max_c[n,c]] += dout[n,c,hh, ww]
    print(dx.shape)
    '''
    for n in range(N):
        for c in range(C):
            for k in range(H_):
                for l in range(W_):
                    pool = x[n,c,k*stride:k*stride+pool_h, l*stride:l*stride+pool_w]
                    maxi = np.max(pool)
                    mask = pool == maxi
                    dx[n,c,k*stride:k*stride+pool_h, l*stride:l*stride+pool_w] += dout[n,c,k,l]*mask
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def svm_loss(x, y):
  """
  Computes the loss and gradient for binary SVM classification.
  Inputs:
  - x: Input data, of shape (N,) where x[i] is the score for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i]
  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  y[y==0]=-1
  #print(y)
  loss = np.mean(np.maximum(0, 1- x*y))
  dx = -y * (1-x*y>0) / y.shape[0]
  
  return loss, dx


def logistic_loss(x, y):
  """
  Computes the loss and gradient for binary classification with logistic 
  regression.
  Inputs:
  - x: Input data, of shape (N,) where x[i] is the logit for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i]
  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  
  
  #case {-1,1}
  y[y==0]=-1
  N=y.shape[0]
  e_negative_y_x = np.exp(-y*x)
  #print('\n\n',x)
  loss = np.mean(np.log(1+e_negative_y_x))
  #print('4r6et7ryrc4h8cru9r\n\n',y)
  dx = ((-y*e_negative_y_x)/(1+e_negative_y_x))/N
  #print(dx)
  return loss, dx
  
  '''
  #case {0,1}
  
  
  sigmoid = 1/(1+np.exp(-x))
  sigmoid=np.transpose(np.vstack((sigmoid,(1-sigmoid))))
  
  N = y.shape[0]
  m = sigmoid[range(N), y]
  log_l = -np.log(m)
  loss = np.sum(log_l) / N
  
  Y_broad = np.zeros(sigmoid.shape)
  Y_broad[range(N), y] = 1
  dx = (sigmoid - Y_broad)/N
  return loss, dx[:,0]
  '''

def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.
  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C
  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  exp_x = np.exp(x)
  div = np.sum(exp_x, axis=1)
  div = div[:, np.newaxis]
  softmax_x = exp_x / div
  
  N = y.shape[0]
  m = softmax_x[range(N), y]
  log_l = -np.log(m)
  loss = np.sum(log_l) / N
  
  Y_broad = np.zeros(x.shape)
  Y_broad[range(N), y] = 1
  dx = (softmax_x - Y_broad)/N
  return loss, dx
