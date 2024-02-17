import pandas as pd
import math
import copy

# K-Nearest Neighbors (KNN) model
# [description] KNN is a non-parametric, lazy learning algorithm that can be used for classification 
# and regression. It is a type of instance-based learning where training is done at query time. 
# The KNN algorithm assumes that similar instances exist in close proximity. KNN uses the distance metric 
# to find the k-nearest neighbors and then uses a kernel as a function of the distance to classify 
# the query point or predict its value.
#
# [input] is training data and labels
# [output] is trained model and logs
#
class KNN:
    def __init__(self):
        self.hyperparameters = None # k for KNN
        self.prediction_type = None # classification or regression
        self.function = None # function learned from training data
        self.positive_class = None # positive class (if binary classification is used)
        
    # Set the hyperparameters for the model (k for KNN)
    def set_params(self, hyperparameters):
        self.hyperparameters = hyperparameters

    # Train the model
    def fit(self, X, y):
        # learning curve: should produce logs of model metric as function of % of trainng data 
        # used for training. There is no learning curve associated with KNN.
        learning_metrics = None
        # deep copy the training data
        data = copy.deepcopy(X)
        # add the target to the data
        data['target'] = y
        # regardless of the prediction type, the function is the data itself
        if self.prediction_type == 'classification':
            self.function = data
        elif self.prediction_type == 'regression':
            self.function = data
        return learning_metrics
    
    # Predict the target values
    def predict(self, X):
        # extract the training data from the function
        data = (self.function).values
        y_pred = []
        # for each query point in X, find its k-nearest neighbors and predict the target value
        for i in range(len(X)):
            query = X.iloc[i].values.tolist()
            k_nearest_neighbors = self.knn(data, query, self.hyperparameters['k'])
            y_pred.append(self.vote(k_nearest_neighbors))
        y_pred = pd.DataFrame(y_pred) # convert to dataframe object
        return y_pred

    # Calculate the Euclidian distance between two points in d-dimensional space where 
    # d is the number of features
    def euclidian_distance(self, p1, p2):
        return sum((p1[i] - p2[i])**2 for i in range(len(p2) - 1))**0.5
    
    # Find the k-nearest neighbors of the query point
    def knn(self, data, query, k):
        distances = []
        # calculate the distance between the query point and each example point in the training data
        for example in data:
            # append the distance and the example point to the distances list
            distances.append((self.euclidian_distance(example, query), example))
        # sort by distance and return the k nearest neighbors
        distances.sort(key=lambda x: x[0])
        k_nearest_neighbors = distances[:k]
        return k_nearest_neighbors
    
    # Gaussian kernel function for weighted average of k-nearest neighbors
    def kernel(self, distance):
        return (1/(math.sqrt(2*math.pi))) * math.exp(-0.5*distance**2)
    
    # Vote for the target value or label using a weighted average of the k-nearest neighbors
    def vote(self, k_nearest_neighbors):
        weighted_sum_of_labels = 0
        sum_of_weights = 0
        for neighbor in k_nearest_neighbors:
            # weight is calculated using a Gaussian kernel
            distance = neighbor[0]
            weight = self.kernel(distance)
            # weight x target value
            weighted_sum_of_labels += weight * neighbor[1][-1]
            # normalization factor
            sum_of_weights += weight
        return weighted_sum_of_labels / sum_of_weights