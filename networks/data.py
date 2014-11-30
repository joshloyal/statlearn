import pandas as pd
import numpy as np

def mean_and_std(X, axis=0):
    _mean = X.mean(axis=axis)
    _std  = X.std(axis=axis)
    
    # make sure we don't scale by 0. (you'll get NaN errors)
    if isinstance(_std, np.ndarray):
        _std[_std == 0.0] = 1.0
    else:
        _std = 1.

    return _mean, _std

def scale_data(data):
    scale = np.amax(data)
    z = data / scale
    return z

def sphere_data(data):
    theta, sigma = mean_and_std(data)
    z = (data - theta) / sigma
    return z

# load the MNIST dataset
def train_MNIST(test_size=0.2):
    # load the MNIST data from csv
    train_csv = 'MNIST_train.csv'
    data = pd.read_csv(train_csv, dtype='float64')
    data = np.array(data)
    X, y = data[:,1:], data[:,0]
    return X, y

def test_MNIST():
    test_csv = 'MNIST_test.csv'
    data = pd.read_csv(test_csv, dtype='float64')
    data = np.array(data) 
    return data
