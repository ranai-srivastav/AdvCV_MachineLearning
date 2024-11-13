import numpy as np
from util import *


# do not include any more libraries here!
# do not put any code outside of functions!


############################## Q 2.1.2 ##############################
# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do XW + b.
# X be [Examples, Dimensions]
def initialize_weights(in_size, out_size, params, name=""):
    W, b = None, None

    ##########################
    ##### your code here #####
    ##########################

    # Weights initialization
    rng = np.random.default_rng()
    val = np.sqrt(6 / (in_size + out_size))
    W = rng.uniform(-val, val, [in_size, out_size])

    # bias initilizations
    b = np.zeros(out_size)

    params["W" + name] = W
    params["b" + name] = b


############################## Q 2.2.1 ##############################
# x is a matrix
# a sigmoid activation function
def sigmoid(x):
    # Samples along rows
    # data along columns
    res = None

    ##########################
    ##### your code here #####
    res = 1 / (1 + np.exp(-x))
    ##########################

    return res


############################## Q 2.2.1 ##############################
def forward(X, params, name="", activation=sigmoid):
    """
    Do a forward pass

    Keyword arguments:
    X -- input vector [Examples x D]
    params -- a dictionary containing parameters
    name -- name of the layer
    activation -- the activation function (default is sigmoid)
    """
    pre_act, post_act = None, None
    # get the layer parameters

    # X is N samples x D
    # W is 2 x 25
    W = params["W" + name]
    # b is [25,]
    b = params["b" + name]

    ##########################
    ##### your code here #####
    # D be the number of datapoints in 1 sample
    # N is the number of samples
    # K is the number of outputs or the number of neurons in the next layer

    # Z | NxK | pre_activation function (WX+b)
    # W | DxK | Weights matrix where for each neuron in layer l, how does it transform to neuron in l+1
    # X | NxD | The input with N samples where each sample has N datapoints
    # B | Kx1 | bias for K neurons in the next layer
    # Y | NxK
    # Z = XW + B = NxD DxK + Kx1 = N samples x K neurons + K x 1

    # Add the bias for the output of Wx for each sample (each row)
    pre_act = X @ W + b
    post_act = activation(pre_act)
    ##########################

    # store the pre-activation and post-activation values
    # these will be important in backprop
    params["cache_" + name] = (X, pre_act, post_act)

    return post_act


############################## Q 2.2.2  ##############################
# x is [examples,classes]
# softmax should be done for each row
def softmax(x):
    res = None

    ##########################
    ##### your code here #####
    row_wise_max = np.max(x, axis=1)
    max_shaped = np.tile(row_wise_max, (x.shape[1], 1)).T
    translated = x - max_shaped
    s_i = np.exp(translated)
    S = np.sum(s_i, axis=1)
    S_shaped = np.tile(S, (x.shape[1], 1)).T
    res = s_i / S_shaped
    ##########################

    return res


############################## Q 2.2.3 ##############################
# compute total loss and accuracy
# y is size [examples,classes]
# probs is size [examples,classes]
def compute_loss_and_acc(y, probs):
    loss, acc = None, None

    ##########################
    ##### your code here #####
    # probs is output of the network
    # y is the training dataset
    loss = -np.sum(np.multiply(y, np.log(probs)))

    # get index of max prob for each row
    compare_class = np.argmax(y, axis=1) - np.argmax(probs, axis=1)
    num_true = compare_class.shape[0] - np.count_nonzero(compare_class)
    acc = num_true / compare_class.shape[0]
    ##########################

    return loss, acc


############################## Q 2.3 ##############################
# we give this to you
# because you proved it
# it's a function of post_act
def sigmoid_deriv(post_act):
    res = post_act * (1.0 - post_act)
    return res


def backwards(delta, params, name="", activation_deriv=sigmoid_deriv):
    """
    Do a backwards pass

    Keyword arguments:
    delta -- errors to backprop
    params -- a dictionary containing parameters
    name -- name of the layer
    activation_deriv -- the derivative of the activation_func
    """
    grad_X, grad_W, grad_b = None, None, None
    # everything you may need for this layer
    W = params["W" + name]
    b = params["b" + name]
    X, pre_act, post_act = params["cache_" + name]

    n = X.shape[0]
    d = X.shape[1]
    k = b.shape[0]

    # x - w_i - z_i - f_i(z_i) - y_i --> w_i+1 - z_i+1 - f_i+1(z_i+1) - y_i+1 -> Loss (y)(log(y_hat))

    # do the derivative through activation first
    # (don't forget activation_deriv is a function of post_act)
    # then compute the derivative W, b, and X
    ##########################
    ##### your code here #####
    # z = n rows x k cols
    # y = n rows x k cols
    # n is the number of samples, d is the number of input neurons, k is the number of output neurons

    # grad_X is L/x =  L/y y/z z/x
    # delL_delX = n row x d cols ? #TODO If thinking per sample
    # delL_delY = n row x k cols
    # delY_delZ = k rows x k cols
    # delZ_delX = k rows x d cols
    delL_delY = delta
    delY_delZ = activation_deriv(post_act)
    delZ_delX = W
    delL_delX = np.multiply(delL_delY, delY_delZ) @ delZ_delX.T
    grad_X = delL_delX

    # grad_W is L/W = L/y y/z z/w will be
    # delL_delW sum over N (N samples deep x d rows x k columns)
    # delL_delY n samples x 1 row x k cols
    # delY_delZ n samples x 1 row x k cols
    # delZ_delW n samples x k rows x d cols
    delL_delY = delta
    delY_delZ = activation_deriv(post_act)
    delZ_delW = X
    delL_delW = np.multiply(delL_delY, delY_delZ).T @ delZ_delW
    grad_W = delL_delW.T

    # grad_b is delL_delB = L/Y Y/Z Z/B #TODO shapes are wrong
    # delL_delB 1 row x k cols
    # delL_delY 1 row x k cols
    # delY_delZ n row x k cols
    # delZ_delB k rows x k cols
    delL_delY = delta
    delY_delZ = activation_deriv(post_act)
    delZ_delB = np.eye(k)
    delL_delB = np.multiply(delL_delY, delY_delZ) @ delZ_delB
    grad_b = np.sum(delL_delB.T, axis=1)
    ##########################

    # store the gradients
    params["grad_W" + name] = grad_W
    params["grad_b" + name] = grad_b
    return grad_X


############################## Q 2.4 ##############################
# split x and y into random batches
# return a list of [(batch1_x,batch1_y)...]
def get_random_batches(x, y, batch_size):
    batches = []
    ##########################
    ##### your code here #####
    i = 0
    idx = np.arange(x.shape[0])
    while i < (x.shape[0] // batch_size):
        chosen_idx = np.random.choice(idx, batch_size, replace=False)
        chosen_x = x[chosen_idx, :]
        chosen_y = y[chosen_idx, :]
        batches.append((chosen_x, chosen_y))
        idx = [a for a in idx if a not in chosen_idx]
        i += 1
    ##########################
    return batches
