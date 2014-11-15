import numpy as np
import theano
import theano.tensor as T
from theano import function

class LogitRegression(object):
    """Single-class Logistic Regression

    The probability tha that an input vector x is a member 
    of class i, can be written as
        P(Y=i|x,W,b) = softmax(W*x + b).

    we use the cross-entropy loss function
        l({W,b}) = -y*log(p_1) - (1-y)*log(1-p_1)
    and a regularized loss
        cost = 1/n * sum_{D}(loss) + lambda * sum(W**2) 
    """
    
    def __init__(self, rate=0.1, reg=0.01, steps=10000):
        self.learning_rate = rate
        self.reg = reg
        self.training_steps = steps

    def fit(self, data_X, data_y):
        n_samples, n_features = data_X.shape
        n_targets = len(np.unique(data_y))
        
        # build the theano model
        self.X = T.matrix('X')
        self.y = T.vector('y')
 
        # initialize weight matrix
        self.W = theano.shared(value=np.random.randn(n_features), name='W')
        
        # initialize bias vector
        self.b = theano.shared(value=0., name='b')

        # prediction probability
        self.p_1 = 1 / (1 + T.exp(-T.dot(self.X,self.W) - self.b))

        # prediciton function
        self.y_pred = self.p_1 > 0.5
        
        # cross-entropy loss
        self.cross_entropy =  -self.y * T.log(self.p_1) - \
                                (1-self.y)*T.log(1-self.p_1)
        
        # cost function
        self.cost = self.cross_entropy.mean() + self.reg * (self.W ** 2).sum()
        
        # gradients wrt the weights and biases
        self.gW, self.gb = T.grad(self.cost, [self.W, self.b])
        
        # perform gradient descent
        gd = theano.function(
                inputs=[self.X, self.y],
                outputs=[self.y_pred, self.cross_entropy],
                updates=((self.W, self.W-self.learning_rate*self.gW), (self.b, self.b - self.learning_rate*self.gb)))

        for i in range(self.training_steps):
            pred, err = gd(data_X, data_y)

        return self

    def predict(self, data_X):
        pred = theano.function(inputs=[self.X], outputs=self.y_pred)
        return pred(data_X)

         
#class LogisticRegression(object):
#    """Multi-class Logistic Regression
#
#    The probability tha that an input vector x is a member 
#    of class i, can be written as
#        P(Y=i|x,W,b) = softmax(W*x + b).
#    We use the negative log-likelihood as our  loss-function:
#        l(theta = {W,b}, D) = - sum_{D} log(P(Y=y|x,W,b) 
#    """
#    def __init__(self, X, n_features, n_targets):
#        """ Initialize the parameters of the logistic regression
#
#        Attributes
#        ----------
#        X : theano.tensor.TensorType (n_samples, n_features)
#            Training vectors, where n_samples is the number of samples
#            and n_features is the number of features
#            
#        n_features : int
#            number of features
#
#        n_targets : int
#            number of targets
#        """
#
#        # initialize weight matrix
#        self.W = theano.shared(value=np.zeros( (n_features, n_targets), 
#                                                dtype=theano.config.floatX ),
#                               name='W',
#                               borrow=True)
#        
#        # initialize bias vector
#        self.b = theano.shared(value=np.zeros( (n_features,), dtype=theano.config.floatX),
#                               name='b',
#                               borrow=True)
#        
#        # symbolic expression for P(Y|x,W,b) = softmax(X*W + b)
#        self.p_y_given_x = T.nnet.softmax(T.dot(X, self.W) + self.b)
#        
#        # class prediction is the class with the largest probability
#        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
#
#        # model parameters
#        self.params = [self.W, self.b]
#
#    def negative_log_likelihood(self, y):
#        # the mean log-likelihood across the data
#        return -T.mean( T.log(self.p_y_given_x)[T.arange(y.shape[0]), y] )
#    
#    def errors(self, y):
#        # T.neq returns a vector of 0s and 1s, where 1 
#        # represents a mistake in prediction
#        return T.mean( T.neq(self.y_pred, y) )

if __name__ == '__main__':
    
    # generate a dataset for 700 gaussian distriubted features
    N = 400
    feats = 700
    D = (np.random.randn(N,feats), np.random.randint(size=N, low=0, high=2))

    Xtrain, Xtest = D[0][:200], D[0][200:]
    ytrain, ytest = D[1][:200], D[1][200:]

    # run the classifier
    classifier = LogitRegression()
    test = classifier.fit(Xtrain, ytrain).predict(Xtest)

    tp = test[test == ytest]
    print len(tp)/200.
