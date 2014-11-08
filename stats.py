def sphere_data(X, cov):
    '''
    Sphere the data, 
    i.e. the common covariance matrix of the transformed data is the identity.

    Inputs:
    X   = n x k matrix (where n = # of observations, k = # of variables)
    cov = pooled covariance matrix of the data set
    '''
    
    m,n = cov.shape
    la, U = linalg.eigh(cov)    # NOTE: cov-matrix is real-hermitian
    Dsqrt = linalg.diagsvd(1./np.sqrt(la),m,n)

    # sphere the data : Xstar = D^(-1/2) * U.T * X
    Xstar = np.apply_along_axis(lambda x: Dsqrt.dot((U.T).dot(x)), 1, X)

    return Xstar


def pooled_covariance(X, y):
    '''
    Estimate the pooled covariance matrix over the data

    Inputs:
    X = n x k matrix (where n = # of observations, k = # of variables)
    y = training labels matrix of true classifications with indices 1 to K
    '''
    
    K = int(max(y) + 1) # the number of classes
    N = y.shape[0] # the number of samples
    
    cov = []
    for k in range(0,K):
        inds = (y == k).nonzero()[0]
        cov.append( covariance_matrix(X[inds], norm=(N-K)) )
    sigmaHat = sum(cov)

    return sigmaHat

def class_priors(y):
    '''
    Estimate the prior probability for each class
    
    Inputs:
    y = training labels matrix of true classifications with indices 1 to K
    '''

    N = y.shape[0] # the number of samples
    prior = [Nk[1]/N for Nk in itemfreq(y)]
    return prior

def class_means(X, y):
    '''
    Estimate the means in each class

    Inputs:
    X = n x k matrix (where n = # of observations, k = # of variables)
    y = training labels matrix of true classifications with indices 1 to K
    '''
    K = int(max(y) + 1) # the number of classes
    
    mean = []
    for k in range(0,K):
        inds = (y == k).nonzero()[0]             # get the indices of class K
        mean.append(np.mean(X[inds], axis=0))    # get the mean of class K

    return mean

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
