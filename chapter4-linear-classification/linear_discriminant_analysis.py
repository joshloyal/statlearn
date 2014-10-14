from statlearn import Point
from scipy.stats import itemfreq
import numpy as np
from sklearn.lda import LDA
import matplotlib.pyplot as plt
from matplotlib import colors

# colormap
cmap = colors.LinearSegmentedColormap(
        'red_blue_classes',
        {'red': [(0, 1, 1), (1, 0.7, 0.7)],
         'green': [(0, 0.7, 0.7), (1, 0.7, 0.7)],
         'blue': [(0, 0.7, 0.7), (1, 1, 1)]})
plt.cm.register_cmap(cmap=cmap)

def linear_discriminant_analysis(Xtrain, Ytrain):
    '''
    Compute classification based on LDA, i.e. assume we can model
    each class density as a multivariate Gaussian with a common covariance matrix

    Inputs:
    Xtrain = training matrix
    Ytrain = training labels matrix of true classifications with indices 1 to K
    '''

    K = int(max(Ytrain) + 1) # the number of classes
    N = Ytrain.shape[0] # the number of samples

    # estimate the prior probability for each class
    prior = [Nk[1]/N for Nk in itemfreq(Ytrain)]
    
    # estimate the mean vector for each class
    mean = []
    for k in range(0,K):
        inds = (Ytrain == k).nonzero()[0]             # get the indices of class K
        mean.append(np.mean(Xtrain[inds], axis=0))    # get the mean of class K
    
    # estimate pooled covariance matrix, i.e. sum the covariance matrix of each class
    cov = []
    for k in range(0,K):
        inds = (Ytrain == k).nonzero()[0]
        cov.append( covariance_matrix(Xtrain[inds], norm=(N-K)) )
    sigmaHat = sum(cov)

def covariance_matrix(X, norm=0):
    '''
    Calculate the covariance matrix of holding ordered sets of row data

    Inputs:
    X = n x k matrix (where n = # of observations, k = # of variables)
    '''
    
    X = np.matrix(X)
    N = X.shape[0]
    K = X.shape[1]

    # calculate a matrix of means
    mean = np.ones((N,N)) * X * (1./N)
    dev  = X - mean # matrix of deviations from the mean
    
    dev2 = dev.T * dev
    if norm == 0:
        cov  = dev2 / N
    else:
        cov = dev2 / norm

    return cov

def scikit_lda(X, y):
    lda = LDA()
    y_pred = lda.fit(X, y, store_covariance=True).predict(X)
    plot_data(lda, X, y, y_pred)

def plot_data(lda, X, y, y_pred):
    tp = (y == y_pred) # True Positive
    tp0, tp1 = tp[y == 0], tp[y == 1]
    X0, X1 = X[y == 0], X[y == 1]
    X0_tp, X0_fp = X0[tp0], X0[~tp0]
    X1_tp, X1_fp = X1[tp1], X1[~tp1]
    xmin, xmax = X[:, 0].min(), X[:, 0].max()
    ymin, ymax = X[:, 1].min(), X[:, 1].max()

    # class 1: dots
    plt.plot(X0_tp[:, 0], X0_tp[:, 1], 'o', color='red')
    plt.plot(X0_fp[:, 0], X0_fp[:, 1], '.', color='#990000')
    
    # class 1: dots
    plt.plot(X1_tp[:, 0], X1_tp[:, 1], 'o', color='blue')
    plt.plot(X1_fp[:, 0], X1_fp[:, 1], '.', color='#000099')

    # class 0 and 1 : areas
    nx, ny = 200, 100
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx),
                         np.linspace(y_min, y_max, ny))
    Z = lda.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = Z[:,1].reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z, cmap='red_blue_classes',
                   norm=colors.Normalize(0., 1.))
    plt.contour(xx, yy, Z, [0.5], linewidths=2., colors='k')

    # means
    plt.plot(lda.means_[0][0], lda.means_[0][1], 'o', color='black', markersize=10)
    plt.plot(lda.means_[1][0], lda.means_[1][1], 'o', color='black', markersize=10)

    plt.show()

def dataset_fixed_cov(n,dim):
    '''
    Generate two Gaussian samples with the same covariance matrix

    Inputs:
    n   = number of samples
    dim = number of features

    Note:
    N(mu, sigma) = sigma*np.random.randn(...) + mu
    ''' 
    cov = np.array( [[0., -0.23], [0.83, 0.23]] )
    X = np.r_[np.dot(np.random.randn(n, dim), cov), 
              np.dot(np.random.randn(n, dim), cov) + np.array([1, 1])]
    y = np.hstack( (np.zeros(n), np.ones(n)) )
    return X, y

if __name__ == '__main__':
    X, y = dataset_fixed_cov(n=300,dim=2) 
    scikit_lda(X, y)
    #linear_discriminant_analysis(X,y)
