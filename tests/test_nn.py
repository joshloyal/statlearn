from statlearn.networks.network import NeuralNetwork
from statlearn.preprocessing.data import sphere_data
from statlearn.utils.activations import sigmoid
from sklearn.datasets import make_blobs
from sklearn.cross_validation import train_test_split
import yaml as yaml
import numpy as np

def make_yaml():
    yaml_init = ''' # Hyper Parameters for Neural Network
    n_epochs : 100
    network :
     shape : [20, 10, 3]
     activ : sigmoid
     parameters : 
      learning : 0.01
      momentum : 0.5
      regularizer : 0.01
    '''    

    return yaml_init

if __name__ == '__main__':
    # import / preprocess data
    X, y = make_blobs(n_samples=1000, n_features=20)
    X = sphere_data(X) 
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2)
    
    # load parameters
    init = make_yaml()
    args = yaml.load(init)

    # train network using stochastic gradient descent
    net = NeuralNetwork(args['network']['shape'], activ=sigmoid, parameters = args['network']['parameters']) 
    net.sgd(train_x, train_y, n_epochs=args['n_epochs'])
    
    # make a prediction
    yhat = net.predict(test_x)
    print 'test accuracy: %f %%'%(np.mean(yhat == test_y)*100)
