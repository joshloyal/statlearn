import numpy as np

def softmax(z):
    """ Numerically stable softmax function """
    maxes = np.amax(z, axis=1)
    maxes = maxes.reshape(-1, 1)
    ep = np.exp(z - maxes)
    z = ep / np.sum(ep, axis=1).reshape(-1,1)

    return z

def identity(z):
    return z

def d_identity(z):
    return 1.0

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def d_sigmoid(z):
    return z*(1.0 - z)

def tanh(z):
    return np.tanh(z)

def d_tanh(z):
    return 1. - z*z

def rectifier(z):
    return np.maximum(0.0, z)

def d_rectifier(z):
    if z < 0.0:
        return 0.0
    elif z > 0.0:
        return 1.0
    else:
        return 0.1
d_vec_rectifier = np.vectorize(d_rectifier)

def softplus(z):
    return np.log(1. + np.exp(z))

def d_softplus(z):
    return sigmoid(z)

def cross_entropy(py_x, y):
    return -np.mean( y*np.log(py_x) )

def d_cross_entropy(py_x, y):
    return py_x - y

derivative = { sigmoid : d_sigmoid,
               tanh : d_tanh,
               rectifier : d_vec_rectifier,
               softplus : d_softplus,
               cross_entropy : d_cross_entropy,
               identity : d_identity }
