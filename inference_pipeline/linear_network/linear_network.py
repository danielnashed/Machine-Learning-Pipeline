import pandas as pd
import copy
import operator
import numpy as np
import sys
from inference_pipeline.decision_tree import node as TreeNode
from inference_pipeline.decision_tree import tree_pruner as TreePruner
from inference_pipeline.decision_tree import tree_visualizer as TreeVisualizer

class LinearNetwork:
    def __init__(self):
        self.hyperparameters = None # k for KNN
        self.prediction_type = None # classification or regression
        self.function = None # function learned from training data
        self.positive_class = None # positive class (if binary classification is used)
        self.num_classes = None # number of classes (if multi-class classification is used)
        self.column_names = None # column names for the dataset
        self.classes = None # unique classes in the training data
        self.learner = None # learner function
        sys.setrecursionlimit(3000) # set recursion limit to 3000 for large datasets

    """
    'set_params' method is responsible for setting the hyperparameters for the model.
    Args:
        hyperparameters (dictionary): hyperparameters for the model
    Returns:
        None
    """
    def set_params(self, hyperparameters):
        self.hyperparameters = hyperparameters

    """
    'fit' method is responsible for training the model.
    Args:
        X (DataFrame): training data
        y (DataFrame): target variable
    Returns:
        learning_metrics (dictionary): logs of model metric as function of % of training data
    """
    def fit(self, X, y):
        # learning curve: should produce logs of model metric as function of % of training data 
        # used for training. No learning curve is produced for decision tree.
        learning_metrics = None
        data = copy.deepcopy(X) # deep copy training data
        data['target'] = y # add target to data
        self.classes = data['target'].unique() # get all unique classes in the target variable
        self.num_classes = len(self.classes) # get number of classes
        # select learner type based on either classification or regression
        if self.prediction_type == 'classification':
            self.learner = self.logistic_regression # use logistic regression for classification
        elif self.prediction_type == 'regression':
            self.learner = self.linear_regression # use linear regression for regression
        # learn the weights using batch gradient descent 
        self.function = self.learner(data)
        return learning_metrics
    

    def linear_regression(self, data):
        num_features = len(data.iloc[0]) - 1 # number of features
        W = np.zeros(num_features) # initialize weights to zero
        learning_rate = self.hyperparameters['learning_rate'] # learning rate
        count = 0 # count number of iterations
        while True:
            count += 1
            print(f'        Iteration: {count}')
            X = data 
            Y = X['target']
            X = X.drop('target', axis=1)
            delta_W = np.dot(X.T, np.dot(X, W) - Y) # calculate gradient
            W -= learning_rate * delta_W / len(data) # update weights
            if count > 1000 or abs(np.sum(delta_W)) < 1e-6:
                break
        return W
    
    def logistic_regression(self, data):
        num_features = len(data.iloc[0]) - 1 # number of features
        W = np.zeros((num_features, self.num_classes)) # initialize weights to zero
        learning_rate = self.hyperparameters['learning_rate'] # learning rate
        count = 0 # count number of iterations
        mu = 0.01
        while True:
            count += 1
            print(f'        Iteration: {count}')
            X = data.drop('target', axis=1)
            Y = pd.get_dummies(data['target']).values # one-hot encoding of target
            logits = np.dot(X, W) # calculate scores
            # Compute probabilities using softmax function
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            delta_W = np.dot(X.T, (probs - Y)) + 2 * mu * W # calculate gradient of cross-entropy loss
            W -= learning_rate * delta_W / len(data) # update weights
            if count > 1000 or  np.linalg.norm(delta_W) < 1e-10:
                break
        return W

    # def logistic_regression(self, data):
    #     num_features = len(data.iloc[0]) - 1 # number of features
    #     W = np.zeros((num_features, self.num_classes)) # initialize weights to zero
    #     learning_rate = self.hyperparameters['learning_rate'] # learning rate
    #     count = 0 # count number of iterations
    #     while True:
    #         count += 1
    #         print(f'        Iteration: {count}')
    #         X = data.drop('target', axis=1)
    #         Y = pd.get_dummies(data['target']).values # one-hot encoding of target
    #         scores = np.dot(X, W) # calculate scores
    #         exp_scores = np.exp(scores) # exponentiate the scores
    #         probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # softmax probabilities
    #         delta_W = np.dot(X.T, (probs - Y)) # calculate gradient of cross-entropy loss
    #         W -= learning_rate * delta_W / len(data) # update weights
    #         if count > 1000 or  np.linalg.norm(delta_W) < 1e-10:
    #             break
    #     return W

    """
    'predict' method is responsible for predicting the target values. The method traverses the decision
    tree to predict the target values for the query points. The method returns the predicted target
    values as a DataFrame object.
    Args:
        X (DataFrame): query points
        function (Node object): function learned from training data (decision tree in this case)
    Returns:
        y_pred (DataFrame): predicted target values
    """
    def predict(self, X, function=None):
        W = self.function
        if self.prediction_type == 'classification':
            logits = np.dot(X, W)  # calculate scores
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))  # exponentiate the scores
            # exp_logits = np.exp(logits)  # exponentiate the scores
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)  # calculate probabilities
            y_pred = np.argmax(probs, axis=1)  # choose the class with the highest probability
        elif self.prediction_type == 'regression':
            y_pred = np.dot(X, W)  # calculate prediction
            # y_pred = []
            # # for each query point in X, make a prediction
            # for i in range(len(X)):
            #     query = X.iloc[i]
            #     prediction = np.dot(W, query)
            #     y_pred.append(prediction)
        y_pred = pd.DataFrame(y_pred) # convert to dataframe object
        return y_pred


