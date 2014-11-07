#!/usr/bin/env python
from matplotlib import pyplot
import numpy as np
from sklearn import neighbors
from matplotlib.colors import ListedColormap

class Point():
    def __init__(self, color, x1, x2):
        self.color = color
        self.x1 = x1
        self.x2 = x2

def generate_points(nsamp=100):
    '''
    Generate data from a combination of multivariate normal distributions.
    '''

    points = []

    n            = 10 # number of samples
    cov          = np.eye(2)
    means_blue   = np.random.multivariate_normal([1,0], cov, n)
    means_orange = np.random.multivariate_normal([0,1], cov, n)

    # generate 100 observations
    N = range(1,nsamp)
    cov = cov / 5
    
    X, y = [], []
    for i in N:
        mu = means_blue[np.random.randint(n)]
        p  = np.random.multivariate_normal(mu, cov, 1)[0]
        X.append([p[0], p[1]])
        y.append([0])
        
        mu = means_orange[np.random.randint(n)]
        p  = np.random.multivariate_normal(mu, cov, 1)[0]
        X.append([p[0], p[1]])
        y.append([1])

    return np.array(X), np.array(y)

def linear_regression():
    x, y = generate_points()
    
    X = np.c_[np.ones(x.shape[0]), x] # add column of 1's
    X = np.matrix(X)
    Y = np.matrix(y)
    
    X_T = X.transpose()
    beta = np.linalg.inv(X_T * X) * X_T * Y

    decision_boundary = 0.5

    # solve for x2 given x1, y, and beta
    f = lambda x: (decision_boundary - beta.item(0) - beta.item(1)*x) / beta.item(2)

    X1 = [x1 for x1 in np.arange(X[:,1].min(), X[:,1].max(), 0.1)]
    X2 = [f(x1) for x1 in X1]
    
    # color code the plane
    xx, yy = np.meshgrid(np.arange(X[:,1].min(), X[:,1].max(), 0.1), np.arange(X[:,2].min(), X[:,2].max(), 0.1))
    Z = beta.item(0) + beta.item(1)*xx + beta.item(2)*yy 
    
    colors = ListedColormap(['#FFFAF5', '#F5F8FF'])
    pyplot.pcolormesh(xx, yy, Z, cmap=colors, vmin=0, vmax=1)
    pyplot.xlim(xx.min(), xx.max())
    pyplot.ylim(yy.min(), yy.max())
    
    # plot data points used to fit the regression
    y = np.array([l[0] for l in y])
    x_blue, x_orange = x[y == 0], x[y == 1]
    pyplot.plot(x_blue[:, 0], x_blue[:, 1], 'o', color='blue')
    pyplot.plot(x_orange[:, 0], x_orange[:, 1], 'o', color='orange')
    
    # plot the regression line
    pyplot.plot(X1, X2, color = 'black')
    pyplot.show()

def nearest_neighbors(k=15):
    x, y = generate_points() 


    X = np.matrix(x)
    Y = np.matrix(y)

    # Create color maps
    colors = ListedColormap(['#FFFAF5', '#F5F8FF'])

    # Create an instance of Neighbors Classifier and fit the data
    clf = neighbors.KNeighborsClassifier(k)
    clf.fit(X, Y)

    # Plot the decision boundary. For that we will assign a color to each
    # point in the mesh [x_min, m_max] x [y_min, y_max]
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    step = 0.2
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))

    # Ravel returns a flattened array and np.c_ concetenates both flattened arrays
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    pyplot.figure()
    pyplot.pcolormesh(xx, yy, Z, cmap=colors)

    # plot also the training points
    pyplot.xlim(xx.min(), xx.max())
    pyplot.ylim(yy.min(), yy.max())

    y = np.array([l[0] for l in y])
    x_blue, x_orange = x[y == 0], x[y == 1]
    pyplot.plot(x_blue[:, 0], x_blue[:, 1], 'o', color='blue')
    pyplot.plot(x_orange[:, 0], x_orange[:, 1], 'o', color='orange')

    pyplot.show()

if __name__ == '__main__':
    #linear_regression()
    nearest_neighbors(k=15)

