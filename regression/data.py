import pandas as pd
import numpy as np
import theano 
from sklearn.cross_validation import train_test_split
from sklearn.decomposition.pca import PCA
from sklearn.preprocessing import StandardScaler

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
    data = pd.read_csv(train_csv, dtype=theano.config.floatX)
    data = np.array(data)
    X, y = data[:,1:], data[:,0]

    # scale / sphere the X data
    sc = StandardScaler()
    X = sc.fit_transform(X)

    # perform PCA to reduce the dimension of the dataset
    pca = PCA(n_components=100)
    X = pca.fit_transform(X)

    # split into training and testing data
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=test_size)
 
    rval = { 'train' : np.c_[train_y, train_x], 'test' : np.c_[test_y, test_x] }
    return rval

def test_MNIST():
    test_csv = 'MNIST_test.csv'
    data = pd.read_csv(test_csv, dtype=theano.config.floatX)
    data = np.array(data)
    
    # scale / sphere the X data
    sc = StandardScaler()
    data = sc.fit_transform(data)

    # perform PCA to reduce the dimension of the dataset
    pca = PCA(n_components=100)
    data = pca.fit_transform(data)
    
    return data
