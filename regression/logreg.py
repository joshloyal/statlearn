import numpy as np
from data import *
import time
from sklearn.preprocessing import LabelBinarizer
import cPickle as pickle
import csv as csv
from sklearn.decomposition.pca import PCA
from sklearn.preprocessing import StandardScaler

def softmax(z):
    """ Numerically stable softmax function """
    maxes = np.amax(z, axis=1)
    maxes = maxes.reshape(-1, 1)
    ep = np.exp(z - maxes)
    z = ep / np.sum(ep, axis=1).reshape(-1,1)

    return z

def rectifier(z):
    """Simple rectifier activation function"""
    return np.maximum(z)


class LogisticClassifier(object):
    def __init__(self, learning_rate=0.01, reg=0., momentum=0.5):
        self.classifier = LogisticRegression(learning_rate, reg, momentum)
        self.pca = None
        self.scaler = None

    def sgd_optimize(self, data, n_epochs, mini_batch_size):
        data = self._preprocess_data(data)
        sgd_optimization(data, self.classifier, n_epochs, mini_batch_size)

    def _preprocess_data(self, data):
        # center data and scale to unit std
        if self.scaler is None:
             self.scaler = StandardScaler()
             data = self.scaler.fit_transform(data)
        else:
            data = self.scaler.transform(data)

        if self.pca is None:
            # use minika's mle to guess appropriate dimension
            self.pca = PCA(n_components='mle')
            data = self.pca.fit_transform(data)
        else:
            data = self.pca.transform(data)

        return data

class LogisticRegression(object):
    def __init__(self, learning_rate=0.01, reg=0., momentum=0.5):
        self.W = None
        self.learning_rate = learning_rate
        self.reg = reg
        self.momentum = momentum
        self.prev_grad = 0.

    def predict_proba(self, X):
        """
        Compute P(Y|X,W). In Logistic Regression this is a softmax function applied
        to a linear transformation of the input matrix
        """
        return softmax(np.dot(X, self.W))

    def predict(self, X):
        """
        Predict the class with the highest probability
        """
        X = np.c_[np.ones(X.shape[0]), X]
        py_x = self.predict_proba(X)
        return np.argmax(py_x, axis=1)
    
    def cross_entropy(self, py_x, y):
        """
        Cross-entropy cost function
        """
        return -np.mean( y*np.log(py_x) )

    def loss(self, X, y):
        """
        Cross-entropy + L2 regularization
        """
        py_x = self.predict_proba(X)
        return self.cross_entropy(py_x, y) - self.reg*np.trace(np.dot(self.W, self.W.T))/self.W.shape[0]

    def gradient(self, X, y):
       py_x = self.predict_proba(X)
       grad = -np.dot(X.T, y - py_x)/X.shape[0] + self.reg*self.W
       return grad
    
    def fit(self, X, y):
        """
        performs one step of gradient descent
        """
        # get the dimensions of our data
        n_samples, n_features = X.shape[0], X.shape[1]+1
        n_targets = len(np.unique(y))

        # add a column to the data matrix to incorporate the bias term
        X = np.c_[np.ones(n_samples), X]
        
        # one-vs-all labeling
        lb = LabelBinarizer()
        y = lb.fit_transform(y)
        
        # initialize the weights 
        if self.W is None:
            self.W = np.zeros( (n_features, n_targets) )
       
        # perform the optimization using gradient descent with momentum
        grad = self.gradient(X,y)
        self.W = self.W - self.learning_rate*(grad + self.momentum*self.prev_grad)
        self.prev_grad = grad

        return self.loss(X,y)

    def save(self, filename):
        pickle.dump({'W':self.W}, open(filename,'wb'), 2)

    def load(self, filename):
        loaded = pickle.load(open(filename, 'rb'))
        self.W = loaded['W']
    

def sgd_optimization(dataset, classifier, n_epochs=10, mini_batch_size=600): 
    # set up parameters
    n_samples = dataset['train'].shape[0]
    n_train_batches = int(np.floor(n_samples / mini_batch_size))

    patience = 5000 # train over at least this many mini-batches 
    patience_increase = 2 # number of mini-batches to increase if we do better
    validation_frequency = min(n_train_batches, patience / 2.)
    improvement_threshold = 0.995
    best_val_error = np.inf
    end_training = False
   

    # loop through mini-batches
    start_time = time.clock()
    for epoch in xrange(n_epochs):
        np.random.shuffle(dataset['train'])
        mini_batches = [
            (dataset['train'][k:k+mini_batch_size, 0], dataset['train'][k:k+mini_batch_size,1:])
            for k in xrange(0, n_samples, mini_batch_size)]
        
        cost = np.zeros(len(mini_batches))
        for i, mini_batch in enumerate(mini_batches):
            cost[i] = classifier.fit(mini_batch[1], mini_batch[0])
            
            # see if we want to check the validation error (NB: i starts at 0)
            iteration = epoch*n_train_batches + i
            if (iteration + 1) % validation_frequency == 0:
                avg_cost = np.mean(cost)
                val_error = 1. - np.mean(classifier.predict(dataset['test'][:,1:]) == dataset['test'][:,0])
                print 'epoch %i, avg. cost %f, validation error %f %%'%(
                            epoch,
                            avg_cost,
                            val_error*100
                        )
                if val_error < best_val_error:
                    if val_error < best_val_error*improvement_threshold:
                        patience = max(patience, iteration*patience_increase)
                    best_val_error = val_error
        
            if patience < iteration:
                end_training = True
                break

        if end_training:
            end_time = time.clock()
            val_error = 1. - np.mean(classifier.predict(dataset['test'][:,1:]) == dataset['test'][:,0])
            print 'Done training: epoch %i, validation error %f %%'%(
                        epoch,
                        val_error*100
                    )
            break


def train():
    data = train_MNIST()
    train_y, train_x = data['train'][:,0], data['train'][:,1:]
    test_y, test_x = data['test'][:,0], data['test'][:,1:]
    classifier = LogisticClassifier()
    X = classifier._preprocess_data(data['train'])
    print X.shape
    #classifier = LogisticRegression()
    #sgd_optimization(data, classifier, n_epochs=5000)
    #classifier.save('pca_test.pkl')

def test():
    data = test_MNIST()
    classifier = LogisticRegression()
    classifier.load('pca_test.pkl')
    yhat = classifier.predict(data)
    predictions_file = open('logreg.csv', 'wb')
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(["ImageId", "Label"])
    open_file_object.writerows( zip(range(1, data.shape[0]+1), yhat) )
    predictions_file.close()

if __name__ == '__main__':
    train()
    #test()
