import numpy as np

class GaussNB(object):
    """
    Gaussian Naive Bayes

    Attributes
    ----------
    classes_ : array, shape (n_classes,)
        labels of each class.

    class_prior_ : array, shape (n_classes,)
        probability of each class.

    class_count_ : array, shape (n_classes,)
        number of training samples observed in each class.

    theta_ : array, shape (n_classes, n_features)
        mean of each feature per class

    sigma_ : array, shape (n_classes, n_features)
        variance of each feature per class
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
        
        epsilon = 1e-9 
         
        # initialize paramters
        self.classes_ = np.unique(y)
        n_features = X.shape[1]
        n_classes  = len(self.classes_)
        self.theta_ = np.zeros((n_classes, n_features))
        self.sigma_ = np.zeros((n_classes, n_features))
        self.class_prior_ = np.zeros(n_classes)
        self.class_count_ = np.zeros(n_classes)
        
        # calculate class priors, class2idx : class -> index
        class2idx = dict((cls,idx) for idx, cls in enumerate(self.classes_))
        for y_i in np.unique(y):
            i = class2idx[y_i]
            X_i = X[y == y_i, :] # observations from class y_i
            N_i = X_i.shape[0]

            self.theta_[i,:], self.sigma_[i,:] = self._calculate_mean_variance(X_i)
            self.class_count_[i] = N_i

        self.sigma_[:,:] += epsilon # in case any sigma_i == 0
        self.class_prior_[:] = self.class_count_ / np.sum(self.class_count_)
        return self

    def predict(self, X):
        """
        Perform classification on an array of test vectors X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        C : array, shape = [n_samples]
            Precited target values for X
        """
        jll = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(jll, axis=1)]
    
    @staticmethod
    def _calculate_mean_variance(X):
        """ Compute Gaussian mean and variance. 
            (NB - each dimension (column) is treated as independent -- you get
            variance not covariance)
        """
        return np.mean(X, axis=0), np.var(X, axis=0)

    def _joint_log_likelihood(self, X):
        joint_log_likelihood = []
        for i in range(np.size(self.classes_)):
            jointi = np.log(self.class_prior_[i])
            n_ij = - 0.5 * np.sum(np.log(2. * np.pi * self.sigma_[i, :]))
            n_ij -= 0.5 * np.sum(((X - self.theta_[i, :]) ** 2) /
                                 (self.sigma_[i, :]), 1)
            joint_log_likelihood.append(jointi + n_ij)

        joint_log_likelihood = np.array(joint_log_likelihood).T
        return joint_log_likelihood
