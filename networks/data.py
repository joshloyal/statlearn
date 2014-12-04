import pandas as pd
import numpy as np

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
