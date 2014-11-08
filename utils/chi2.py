import numpy as np
from scipy.stats import chi2

class Data():
    def __init__(self, data, error):
        self.data  = data
        self.error = error

def weighted_mean(X):
    num, dem = 0, 0
    for x in X:
        num += (x.data/x.error**2)
        dem += (1/x.error**2)
    return num/dem

def weighted_error(X):
    err = 0
    for x in X:
        err += (1/x.error**2)
    return 1/err

def chi2_test():
    X = [ Data(91.161, 0.013), 
          Data(91.174, 0.011),
          Data(91.186, 0.013),
          Data(91.188, 0.013) ]
    
    Mz  = weighted_mean(X)
    err = weighted_error(X)
    
    X2, ndf = 0., 3.
    for x in X:
        X2 += (x.data - Mz)**2 / x.error**2
    
    pval = 1 - chi2.cdf(X2, ndf)
    print (X2, pval)

if __name__ == '__main__':
    chi2_test()


 
