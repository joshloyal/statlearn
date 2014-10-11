from matplotlib import pyplot
import numpy as np
from matplotlib.colors import ListedColormap

labels = ['blue', 'orange']

class Point():
    def __init__(self, color, x1, x2):
        self.color = color
        self.x1 = x1
        self.x2 = x2

def generate_points():
    '''
    Generate data from a combination of multivariate normal distributions.
    '''

    points = []

    n            = 10 # number of samples
    cov          = np.eye(2)
    means_blue   = np.random.multivariate_normal([1,0], cov, n)
    means_orange = np.random.multivariate_normal([0,1], cov, n)

    # generate 100 observations
    N = range(1,100)
    cov = cov / 5

    for i in N:
        mu = means_blue[np.random.randint(n)]
        p  = np.random.multivariate_normal(mu, cov, 1)[0]
        points.append( Point(labels[0], p[0], p[1]) )
        
        mu = means_orange[np.random.randint(n)]
        p  = np.random.multivariate_normal(mu, cov, 1)[0]
        points.append( Point(labels[1], p[0], p[1]) )

    return points

def linear_regression():
    points = generate_points() 

    X = []
    Y = []
    for p in points:
        X.append([1.0, p.x1, p.x2])
        Y.append([1 if p.color == 'blue' else 0])

    X = np.matrix(X)
    Y = np.matrix(Y)

    X_T = X.transpose()
    beta = np.linalg.inv(X_T * X) * X_T * Y

    decision_boundary = 0.5

    # solve for x2 given x1, y, and beta
    f = lambda x: (decision_boundary - beta.item(0) - beta.item(1)*x) / beta.item(2)

    X1 = [x1 for x1 in np.arange(X[:,1].min(), X[:,1].max(), 0.1)]
    X2 = [f(x1) for x1 in X1]
    
    # color code the remaining points
    colors = ListedColormap(['#FFFAF5', '#F5F8FF'])
    xx, yy = np.meshgrid(np.arange(X[:,1].min(), X[:,1].max(), 0.1), np.arange(X[:,2].min(), X[:,2].max(), 0.1))
    Z = beta.item(0) + beta.item(1)*xx + beta.item(2)*yy 
    pyplot.pcolormesh(xx, yy, Z, cmap=colors, vmin=0, vmax=1)
    pyplot.xlim(xx.min(), xx.max())
    pyplot.ylim(yy.min(), yy.max())
    
    for p in points:
        pyplot.plot(p.x1, p.x2, 'o', mfc = 'none',  mec = p.color)

    pyplot.plot(X1, X2, color = 'black')


    pyplot.show()

if __name__ == '__main__':
    linear_regression()

