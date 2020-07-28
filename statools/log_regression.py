import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score


class LogRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, learn_rate, epochs, verbose=10):
        self.learn_rate = learn_rate
        self.epochs = epochs
        self.verbose = verbose

    def fit(self, X, Y, val_data=None):
        self.n = X.shape[0]
        self.m = X.shape[1]

        # Initialize weights, bias
        self.Weights = np.zeros((self.n, 1))
        self.bias = 0

        # Store scores across epochs
        self.train_losses = []
        self.train_scores = []
        self.val_losses = []
        self.val_scores = []
        
        # Iterate through the epochs
        for i in range(1, self.epochs + 1):
            # Forward/Back propagation one epoch
            loss = self.__propagate(X, Y)

            # Get present training accuracy
            accuracy = accuracy_score(Y.flatten(), 
                                      self.predict(X).flatten())

            # Store training scores
            self.train_losses.append(loss)
            self.train_scores.append(accuracy)

            # If validation data presented, get current scores on
            # it too
            if val_data is not None:
                val_accuracy = accuracy_score(val_data[1].flatten(), 
                                    self.predict(val_data[0]).flatten())
                self.val_scores.append(val_accuracy)
                val_loss = self.__loss(val_data[1], 
                                       self.predict(val_data[0]))
                self.val_losses.append(val_loss)

            # If verbose > 0, print every verbose epochs
            if self.verbose > 0 and i % self.verbose == 0:
                print(f'Epoch {i} Loss {loss} Accuracy {accuracy * 100}%')
                if val_data is not None:
                    print(f'Validation Accuracy {val_accuracy * 100}%')
                    
        return self
                
    def predict(self, X, y=None):
        m = X.shape[1]
        Y_prediction = np.zeros((1,m))

        A = self.__sigmoid(np.dot(self.Weights.T, X) + self.bias)
        for i in range(m):
            Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0
        return Y_prediction

    def __propagate(self, X, Y):
        # Forward propagate
        A = self.__sigmoid(np.dot(self.Weights.T, X) + self.bias)
        cost = self.__loss(Y, A)

        # Back propagate
        dw = 1/self.m * np.dot(X, (A - Y).T)
        db = 1/self.m * np.sum(A - Y)

        # Update parameters
        self.Weights = self.Weights - self.learn_rate * dw
        self.bias = self.bias - self.learn_rate * db
        return cost

    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def __loss(self, Y, A):
        cost = -1/self.m * np.sum(np.dot(Y, np.log(A).T) + np.dot((1-Y), 
                                  np.log(1 - A).T))
        return np.squeeze(cost)


if __name__ == '__main__':
    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt

    # Make some synthetic data, two classes
    np.random.seed(0)
    X, y = make_blobs(200, n_features=2, centers=2)

    # Reshape data
    X_train = X.T
    y_train = y.reshape(1, -1)

    # try our classifier!
    clf = LogRegression(.01, 100, verbose=1)
    clf.fit(X_train, y_train)

    # Get and print final accuracy
    y_pred = clf.predict(X_train)
    acc = accuracy_score(y_train.flatten(), 
                         y_pred.flatten())
    print(f'Final accuracy {acc * 100} %')

    # plot scores over epochs
    plt.plot(clf.train_losses, label='loss')
    plt.plot(clf.train_scores, label='accuracy')
    plt.title('Training Scores')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()

    plt.scatter(X_train[0], X_train[1], c=y_train, alpha=.5)
    plt.scatter(X_train[0], X_train[1], c=y_pred, alpha=.5)
    plt.show()
