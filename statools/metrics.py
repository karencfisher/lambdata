import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.exceptions import NotFittedError
import matplotlib.pyplot as plt


class ConfusionMatrix():
    '''
    Constructs a confusion matrix and provides metrics (precision,
    recall, and F1 scores for each class). Metrics implemented for
    multinomial classes.

    Constructor takes y_true and y_predict for arguments, and 
    generates a confusion matrix.
    '''

    def __init__(self, y_true, y_predict, labels=None):
        '''
        Constructor

        Arguments:
            y_true -- ground truth
            y_predict -- model output
            labels -- class labels (optional)

        Exceptions:
            None
        '''
        self.y_true = y_true
        self.y_predict = y_predict

        if labels == None:
            # Get all unique class labels in each y_true and y_predict,
            #  even if not in both sets
            self.labels_ = list(set(y_true).union(set(y_predict)))
        else:
            self.labels_ = labels

        # Generate the confusion and matrix as instance attribute
        self.confusion_matrix = confusion_matrix(self.y_true,
                                                 self.y_predict,
                                                 self.labels_)
        

    def get_confusion_matrix(self):
        '''
        Outputs a Pandas DataFrame formatting the matrix nicely

        Arguments:
            None

        Returns:
            Pandas DataFrame
        '''
        return pd.DataFrame(self.confusion_matrix,
                            columns=self.labels_,
                            index=self.labels_)


    def get_metrics(self):
        '''
        Gets precision, recall, and F1 scores for each class.

        Returns:
            Pandas DataFrame nicely formatting the resultant
            matrix. (Also retains a more raw version as an 
            instance attribute as ConfusionMatrix.metrics_.)
        '''
        cm = self.confusion_matrix

        # Get sums of both axes
        pred_sums = np.sum(cm, axis=0)
        actual_sums = np.sum(cm, axis=1)

        # Calculate metrics for each class. This allows scores
        # for multiple classes
        self.metrics_ = {}
        for i in range(cm.shape[0]):
            precision = cm[i][i] / pred_sums[i]
            recall = cm[i][i] / actual_sums[i]
            f1 = 2 * precision * recall / (precision + recall)
            self.metrics_[self.labels_[i]] = {'precision': precision,
                                'recall': recall, 
                                'f1-score': f1}

        # Return a formatted version of the metrics for each class
        return pd.DataFrame(self.metrics_).T



class MultiClassROC():
    '''
    Calculates ROC curves for multiple classes.

    Constructor takes as arguments the estimator/model, X,
    y, and optionally a list of the class labels. It not 
    specified we will attempt to get them from the model 

    Raises ValueError or NotFittedError if the model has not been 
    fit yet.
    '''

    def __init__(self, model, X, y, classes=None):
        '''
        Constructor

        Arguments:
            model - fitted model
            X, y - X and y values as arrays/lists/Pandas Series

        Exceptions:
            ValueError or NotFittedError if model has not been
            fit.
        '''
        # Validate user input, as well as getting classes
        if classes == None:
            try:
                self.classes_ = model.classes_
            except ValueError:
                raise ValueError('Has model been fit?')
        else:
            self.classes_ = classes

        self.X = X
        self.y = y
        self.model = model

        # Predict probabilities for classes, raising exception
        # if model has not yet been fit
        try:
            self.y_hat = model.predict_proba(X)
        except NotFittedError:
            raise NotFittedError('Has model been fit?')

        self.ROC_ = None


    def fit_ROC(self):
        '''
        'Fit' ROC curves to the various classes.

        Returns:
            Dictionary containing true and false positive rates
            at each threshold.
        '''
        self.ROC_ = {}

        # Isolate each class as if binomial, True/False, and use
        # sklearn.metrics.roc_curve method.
        for i, label in enumerate(self.classes_):
            y = self.y == label
            fpr, tpr, _ = roc_curve(y, self.y_hat[:,i])
            self.ROC_[label] = [fpr, tpr]

        return self.ROC_


    def plot_ROC(self):
        '''
        Plots the ROC curves for classes

        Returns:
            Matplotlib figure object

        Exceptions:
            ValueError if curves have not yet been 'fit.'
        '''
        if self.ROC_ == None:
            raise ValueError('Call fit_ROC method first.')

        self.figure_ = plt.figure(figsize=(10,7))
        for label in self.classes_:
            plt.plot(self.ROC_[label][0],
                     self.ROC_[label][1], 
                     '-o', label=label)

        plt.legend()
        plt.title('ROC Curves')
        plt.xlabel('False positive rates')
        plt.ylabel('True positive rates')
        plt.show()

        return self.figure_
    

        
if __name__ == '__main__':

    # Implement a basic unit test.

    from sklearn.datasets import load_iris
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from IPython.display import display

    # Load the Iris data set from sklearn
    print('Loading iris data...')
    labels = ['setosa', 'versicolor', 'virginica']
    iris = load_iris()
    X = iris.data[:, :2]
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, 
                                                        random_state=42)
    
    # Fit a decision tree to it and predict test data
    print('Building a decision tree...')
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    print(f'Accuracy = {model.score(X_test, y_test) * 100} %')

    # Try confusion matrix object
    print('\nGenerate confusion matrix...')
    confusion = ConfusionMatrix(y_test, y_predict)
    display(confusion.get_confusion_matrix())
    print('\nGenerate metrics...')
    display(confusion.get_metrics())

    # Try the ROC plot
    print('\nGenerate ROC curves...')
    roc = MultiClassROC(model, X_test, y_test)
    roc.fit_ROC()
    roc.plot_ROC()

    print('Success')




