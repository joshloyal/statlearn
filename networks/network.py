import numpy as np
from data import *
from sklearn.cross_validation import train_test_split
from sklearn.datasets import make_blobs
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition.pca import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import time as time
import yaml
import cPickle as pickle
import csv as csv

def softmax(z):
    """ Numerically stable softmax function """
    maxes = np.amax(z, axis=1)
    maxes = maxes.reshape(-1, 1)
    ep = np.exp(z - maxes)
    z = ep / np.sum(ep, axis=1).reshape(-1,1)

    return z

def identity(z):
    return z

def d_identity(z):
    return 1.0

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def d_sigmoid(z):
    return z*(1.0 - z)

def tanh(z):
    return np.tanh(z)

def d_tanh(z):
    return 1. - z*z

def rectifier(z):
    return np.maximum(0.0, z)

def d_rectifier(z):
    if z < 0.0:
        return 0.0
    elif z > 0.0:
        return 1.0
    else:
        return 0.1
d_vec_rectifier = np.vectorize(d_rectifier)

def softplus(z):
    return np.log(1. + np.exp(z))

def d_softplus(z):
    return sigmoid(z)

def cross_entropy(py_x, y):
    return -np.mean( y*np.log(py_x) )

def d_cross_entropy(py_x, y):
    return py_x - y

derivative = { sigmoid : d_sigmoid,
               tanh : d_tanh,
               rectifier : d_vec_rectifier,
               softplus : d_softplus,
               cross_entropy : d_cross_entropy,
               identity : d_identity }

class Layer(object):
    """ Neural Network """
    def __init__(self, shape, activ, parameters = None):
        super(Layer, self).__init__()
        scale=0.1
        self.W = np.random.randn(shape[0], shape[1])*scale
        self._prev_W = np.zeros(shape)
        self.b = np.random.randn(shape[1])*scale
        self._prev_b = np.zeros((shape[1],))
        self.activ = activ
        if parameters is None:
            self.parameters = {'learning' : 0.01, 'momentum' : 0.5, 'regularizer' : 0}
        else:
            self.parameters = parameters


    def forward_prop(self, prev_a):
        self.prev_a = prev_a
        self.activations = self.activ(np.dot(self.prev_a,self.W) + self.b)
        return self.activations

    def back_prop(self, grad): 
        self.delta = grad * derivative[self.activ](self.activations) # (n_sample, n_out)
        self.delta_W = np.dot(self.prev_a.T, self.delta) # (n_in, n_out)
        self.delta_b = np.sum(self.delta, axis=0) # (,n_out)
        del self.prev_a
        del self.activations
        return np.dot(self.delta, self.W.T) # (n_sample, n_in)

    def update(self):
        self._prev_W = self.parameters['momentum']*self._prev_W - self.parameters['learning']*(self.delta_W + self.parameters['regularizer']*self.W)   
        self._prev_b = self.parameters['momentum']*self._prev_b - self.parameters['learning']*self.delta_b
        self.W += self._prev_W
        self.b += self._prev_b

class NeuralNetwork(object):
    def __init__(self, sizes, activ, parameters = None):
        super(NeuralNetwork, self).__init__()

        # build hidden layers 
        self.layers = [Layer((nin, nout), activ = activ, parameters = parameters) for nin,nout in zip(sizes[:-2], sizes[1:])]

        # append a linear output layer (to get probabilites just apply softmax)
        self.layers.append( Layer((sizes[-2],sizes[-1]), activ = identity, parameters = parameters) )

    def feedforward(self, X):
        for layer in self.layers:
            X = layer.forward_prop(X)
        return X

    def predict_proba(self, X):
        z = self.feedforward(X)
        return softmax(z)

    def predict(self, X):
        py_x = self.predict_proba(X)
        return np.argmax(py_x, axis=1)

    def fit(self, X, y):
        lb = LabelBinarizer()
        y_ = lb.fit_transform(y)
        py_x = self.predict_proba(X)
        grad = derivative[cross_entropy](py_x, y_)
        for layer in self.layers[::-1]:
            grad = layer.back_prop(grad)
            layer.update()
        py_x = self.predict_proba(X)
        return cross_entropy(py_x, y_)

    def sgd(self, X, y, n_epochs=10, mini_batch_size=600, test_size=0.2, filename=None):
        sgd_optimization(X, y, self, n_epochs=n_epochs, mini_batch_size=mini_batch_size, test_size=test_size, filename=filename)

    def save(self, filename):
        pickle.dump(self, open(filename, 'wb'), 2)

def combined_dataset(X,y):
    data = np.c_[y,X]
    return data

def sgd_optimization(X, y, classifier, n_epochs=10, mini_batch_size=600, test_size=0.2, filename=None):
    # split into training and testing set
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=test_size)

    # flatten the training and testing data
    train = combined_dataset(train_x, train_y)
    test  = combined_dataset(test_x, test_y)
    
    # parameters
    n_samples = train.shape[0]
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
        np.random.shuffle(train)
        mini_batches = [
            (train[k:k+mini_batch_size, 0], train[k:k+mini_batch_size,1:])
            for k in xrange(0, n_samples, mini_batch_size)]
        
        cost = np.zeros(len(mini_batches))
        for i, mini_batch in enumerate(mini_batches):
            cost[i] = classifier.fit(mini_batch[1], mini_batch[0])
            
            # see if we want to check the validation error (NB: i starts at 0)
            iteration = epoch*n_train_batches + i
            if (iteration + 1) % validation_frequency == 0:
                avg_cost = np.mean(cost)
                val_error = 1. - np.mean(classifier.predict(test[:,1:]) == test[:,0])
                print 'epoch %i, avg. cost %f, validation error %f %%'%(
                            epoch,
                            avg_cost,
                            val_error*100
                        )
                if filename:
                    outfile = filename+'_epoch%i_err%.2f.pkl'%(epoch, val_error*100)
                    classifier.save(outfile)
                if val_error < best_val_error:
                    if val_error < best_val_error*improvement_threshold:
                        patience = max(patience, iteration*patience_increase)
                    best_val_error = val_error
        
            if patience < iteration:
                end_training = True
                break

        if end_training:
            end_time = time.clock()
            val_error = 1. - np.mean(classifier.predict(test[:,1:]) == test[:,0])
            print 'Done training: epoch %i, validation error %f %%'%(
                        epoch,
                        val_error*100
                    )
            filename = 'final_err%.2f.pkl'%(val_error*100)
            classifier.save(filename)
            break

if __name__ == '__main__':
    
    with open('init.yaml', 'rb') as f:
        args = yaml.load(f)
    #X, y = make_blobs(n_samples=1000, n_features=20)
    #net = NeuralNetwork([20, 10, 3], activ=sigmoid, parameters = args['parameters']) 
    #net.sgd(X, y, n_epochs=args['n_epochs'], filename='blobs_net')
    
    # preprocess data
    X, y = train_MNIST()
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    pca = PCA(n_components=args['pca'])
    X = pca.fit_transform(X)
    pickle.dump(scaler, open('MNIST_Scaler.pkl', 'wb'), 2)
    pickle.dump(pca, open('MNIST_pca.pkl', 'wb'), 2)
    
    net = NeuralNetwork(args['network']['shape'], activ=sigmoid, parameters = args['network']['parameters']) 
    net.sgd(X,y, n_epochs=args['n_epochs'], filename='MNIST_NN')

    #X = test_MNIST()
    #scaler = pickle.load(open('MNIST_Scaler.pkl', 'rb'))
    #pca = pickle.load(open('MNIST_pca.pkl', 'rb'))
    #X = scaler.transform(X)
    #X = pca.transform(X)
    #net = pickle.load(open('MNIST_NN_epoch16_err4.79.pkl', 'rb'))
    #yhat = net.predict(X)
    #predictions_file = open('net.csv', 'wb')
    #open_file_object = csv.writer(predictions_file)
    #open_file_object.writerow(["ImageId", "Label"])
    #open_file_object.writerows( zip(range(1, X.shape[0]+1), yhat) )
    #predictions_file.close()
