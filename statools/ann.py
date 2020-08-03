import numpy as np
from sklearn.metrics import accuracy_score


'''
-------------- Activation Functions ----------------------
Each object has two methods:
    g: The activation function
    dg: The partial derivative
'''

class Sigmoid():
    def g(self, z):
        return 1 / (1 + np.exp(-z))
    
    def dg(self, z):
        self.g(z) * (1 - self.g(z))

    
class Tanh():
    def g(self, z):
        return np.tanh(z)

    def dg(self, z):
        return 1 - np.power(np.tanh(z), 2)


class Relu():
    def g(self, z):
        return np.maximum(z, 0)

    def dg(self, z):
        return (z >= 0) * 1


'''
--------------------- Loss Functions ---------------------
'''

def binary_crossentropy(A, Y):
    m = Y.shape[1]
    cost = -1/m * np.sum(np.dot(Y, np.log(A).T) + np.dot((1-Y), 
                         np.log(1 - A).T))
    return np.squeeze(cost)


def categorical_crossentropy(A, Y):
    pass


class Layer():
    def __init__(self, units, activation, learning_rate, 
                 input_shape=None, input_layer=None, 
                 output_layer=None):
        self.units = units
        self.activation = activation()
        self.learning_rate = learning_rate
        self.input_shape = input_shape
        self.input_layer = input_layer
        self.output_layer = output_layer

        self.m = input_shape[1]
        self.Weights = np.random.randn(self.units, 
                                       self.input_shape[0]) * 0.01
        self.bias = np.zeros((self.units, 1))
    
    def forward(self, X):
        self.X = X
        self.Z = np.dot(self.Weights, self.X) + self.bias
        self.A = self.activation.g(self.Z)
        if self.output_layer is not None:
            self.output_layer.forward(self.A)
        return

    def back(self, dZ):
        if self.output_layer is None:
            self.dZ = dZ
        else:
            self.dZ = np.dot(self.output_layer.prev_weights.T, dZ)
            self.dZ *= self.activation.dg(self.Z)

        self.dW = 1/self.m * np.dot(self.dZ, self.X.T)
        self.db = 1/self.m * np.sum(self.dZ, axis=1, keepdims=True)

        self.prev_weights = np.copy(self.Weights)
        self.Weights -= self.learning_rate * self.dW
        self.bias -= self.learning_rate * self.db

        if self.input_layer is not None:
            self.input_layer.back(self.dZ)
        return
        

class ANN():
    '''
    ANN Classifier

    '''
    def __init__(self, input_shape, learning_rate, loss):
        '''
        Args:
            input_shape: The dimensions of the input data to the network.
                         N x M matrix.
            learning_rate: The learning rate for training
            loss: Loss function
        '''
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.loss = loss
        self.layers = []

    def add_layer(self, units, activation):
        '''
        Add a layer to the neural network.

        Args:
            units: the number of neurons in the layer. Integer.
            activation: the activation function, as one of the
                        activation objects (Sigmoid, Tanh, Relu)
        '''
        if len(self.layers) > 0:
            # Not first layer so get pointer to previous layer
            input_layer = self.layers[-1]

            # Get output shape of previous layer to provide
            # input_shape for new layer
            input_shape = (self.layers[-1].units, 
                           self.input_shape[1])
            
            # Instantiate new layer
            layer = Layer(units, activation, self.learning_rate,
                               input_shape=input_shape,
                               input_layer=input_layer)
            
            # Update output_layer pointer of previous layer
            self.layers[-1].output_layer = layer
        else:
            # Instantiate first layer
            layer = Layer(units, activation, self.learning_rate,
                               input_shape=self.input_shape)

        self.layers.append(layer)

    def fit(self, X, y, epochs, verbose=None):
        self.losses = []
        self.accuracies = []

        for i in range(1, epochs + 1):
            # Feed forward getting final output
            self.layers[0].forward(X)
            A = self.layers[-1].A

            # Map outputs
            if self.layers[-1].units > 1:
                y_hat = [np.argmax(A[j]) for j in range(len(A))]
            else:
                y_hat = (A > 0.5) * 1

            # Calculate loss and accuracy
            loss = self.loss(A, y)
            self.losses.append(loss)
            accuracy = accuracy_score(y[0], y_hat[0])
            self.accuracies.append(accuracy)

            # If verbose, print info
            if verbose is not None:
                if i % verbose == 0:
                    print(f'Epoch {i} Loss {loss} Accuracy {accuracy}')

            # Begin back propagation
            dZ = A - y
            self.layers[-1].back(dZ)   

    def predict(self, X):
        # Feed forward
        self.layers[0].forward(X)

        # Get output
        A = self.layers[-1].A

        # Map to output class
        if self.layers[-1].units > 1:
            y_hat = [np.argmax(A[i]) for i in range(len(A))]
        else:
            y_hat = (A > 0.5)
        
        return y_hat



if __name__ == '__main__':
    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt

    # Make some synthetic data, two classes
    np.random.seed(0)
    X, y = make_blobs(200, n_features=2, centers=2)

    # Reshape data
    X_train = X.T
    y_train = y.reshape(1, -1)

    clf = ANN(X_train.shape, 0.01, binary_crossentropy)
    clf.add_layer(5, Tanh)
    clf.add_layer(1, Sigmoid)

    clf.fit(X_train, y_train, 1000, verbose=100)

    # plot scores over epochs
    plt.plot(clf.losses, label='loss')
    plt.plot(clf.accuracies, label='accuracy')
    plt.title('Training Scores')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()
