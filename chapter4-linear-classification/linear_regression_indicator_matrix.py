from statlearn import Point
from sklearn import linear_model
import numpy as np
from matplotlib import pyplot

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
    
def generate_points(N):
    points = []
    
    cov = np.eye(2) / 500
    means  = [ [0.25, 0.25], [0.5, 0.5], [0.75, 0.75] ]
    colors = [ 'orange', 'blue', 'green' ]

    for i in range(1, N):
        for mu, color in zip(means, colors):
            p = np.random.multivariate_normal(mu, cov, 1)[0]
            points.append( Point(color, p[0], p[1]) )
        
    return points

def linreg_indmat():
    points = generate_points(50)

    X = []
    Y = []
    color_dic = { 'blue' : 0, 'orange' : 1, 'green' : 2 }
    for p in points:
        X.append([p.x1, p.x2])
        Y.append(color_dic[p.color])
    Bhat, yHat, gHat, err_frac = linear_regression_indicator_matrix(X,Y)
    print 'Training Error: %.2f' %(err_frac)


    # plot the coeffiecients of the training data for each point 
    X1 = [x[0] for x in X]
    fig = pyplot.figure()
    ax = fig.add_subplot(111) 
    pyplot.ylim(-0.5,1.25)
    ax.plot(X1, np.zeros(len(X1)) - 0.5, 'k+', ms=20)
    ax.plot(X1, yHat[:,0], 'o', mfc = 'none', mec = 'blue')
    ax.plot(X1, yHat[:,1], 'o', mfc = 'none', mec = 'orange')
    ax.plot(X1, yHat[:,2], 'o', mfc = 'none', mec = 'green')
    pyplot.show()

if __name__ == '__main__':
    linreg_indmat()
