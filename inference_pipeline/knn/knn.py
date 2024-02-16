import pandas as pd
import math
import copy

class KNN:
    def __init__(self):
        self.hyperparameters = None
        self.prediction_type = None
        self.function = None
        self.positive_class = None # positive class (if binary classification is used)
        
    def set_params(self, hyperparameters):
        self.hyperparameters = hyperparameters

    def fit(self, X, y):
        # learning curve: should produce logs of model metric as function of % of trainng data used for training
        learning_metrics = None
        data = copy.deepcopy(X)
        data['target'] = y
        if self.prediction_type == 'classification':
            self.function = data
        elif self.prediction_type == 'regression':
            self.function = data
        return learning_metrics
    
    def predict(self, X):
        data = (self.function).values
        y_pred = []
        for i in range(len(X)):
            query = X.iloc[i].values.tolist()
            k_nearest_neighbors = self.knn(data, query, self.hyperparameters['k'])
            y_pred.append(self.vote(k_nearest_neighbors))
        y_pred = pd.DataFrame(y_pred) # convert to dataframe object
        return y_pred

    def euclidian_distance(self, p1, p2):
        return sum((p1[i] - p2[i])**2 for i in range(len(p2) - 1))**0.5
    
    def knn(self, data, query, k):
        distances = []
        for example in data:
            distances.append((self.euclidian_distance(example, query), example))
            # sort by distance and get the k nearest neighbors
        distances.sort(key=lambda x: x[0])
        k_nearest_neighbors = distances[:k]
        return k_nearest_neighbors
    
    def kernel(self, distance):
        return (1/(math.sqrt(2*math.pi))) * math.exp(-0.5*distance**2)
    
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