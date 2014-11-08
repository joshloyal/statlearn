import numpy as np
from sklearn import neighbors
from .utils.point import Point

__all__ = ['linear_regression_indicator_matrix',
           'linear_regression', 'nearest_neighbors']

def linear_regression_indicator_matrix(Xtrain, Ytrain):
    '''
    Compute classification based on linear regression

    Inputs:
    Xtrain = training matrix (without column of ones needed to represent the offset
    Ytrain  = training labels matrix of true classifications with indices 1 - K
    (K is the number of classes)

    '''
    
    K = max(Ytrain) + 1 # the number of classes
    N = len(Ytrain)     # the number of samples

    # form the indicator response matrix
    Y = np.zeros([N, K])
    for i, y in enumerate(Ytrain):
        Y[i][y] = 1.
    
    # append a column of ones to the Xtrain matrix
    X = np.matrix( Xtrain )
    X = np.insert(X, 0, 1, axis=1)

    # calculate the coefficients
    X_T = X.transpose()
    Bhat = np.linalg.inv(X_T * X) * X_T * Y
    Yhat = X*Bhat          # discriminant predictions on the training data
    gHat = Yhat.argmax(1)  # classify this data

    # calculate the training error rate
    err = 0.
    for i, ghat in enumerate(gHat):
        if ghat != Ytrain[i]:
            err += 1.
    err_frac = err / N

    return np.array(Bhat), np.array(Yhat), np.array(gHat), err_frac
    

def linear_regression(x, y): 
    X = np.c_[np.ones(x.shape[0]), x] # add column of 1's
    X = np.matrix(X)
    Y = np.matrix(y)
    
    X_T = X.transpose()
    beta = np.linalg.inv(X_T * X) * X_T * Y
    
    return beta


def nearest_neighbors(x, y, k=15):
 
    X = np.matrix(x)
    Y = np.matrix(y)

    # Create an instance of Neighbors Classifier and fit the data
    clf = neighbors.KNeighborsClassifier(k)
    clf.fit(X, Y)

    return clf
