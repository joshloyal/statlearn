import numpy as np
import cPickle as pickle
from ..utils.activations import *
from ..optimization.sgd import sgd_optimization
from sklearn.preprocessing import LabelBinarizer

class Layer(object):
    """ Single Layer for a Neural Network """
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
