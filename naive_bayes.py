from sklearn import datasets
iris = datasets.load_iris()

class GaussNB(object):
    """
    Gaussian Naive Bayes

    Attributes
    ----------
    """

    def fit(self, X, y):
    """Fit Gaussian Naive Bayes according to X, y

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Training vectors, where n_samples is the number of samples
        and n_features is the number of features.

    y : array-like, shape (n_samples,)
        Target values.

    Returns
    -------
    self : object
        Returns self
    """
    
    # get starting estimates of the probability distributions
    n_features = X.shape[1]
    n_classes  = np.unique(y)
    self.theta = np.zeros((n_classes, n_features))
    self.sigma = np.zeros((n_classes, n_features))
    self.class_prior = np.zeros(n_classes)
    self.class_count = np.zeros(n_classes)

    for y_i in np.unique(y):
        X_i = X[y 

    def predict(self):
