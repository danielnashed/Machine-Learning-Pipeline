import pandas as pd

# Null model
# [description] this class is used as a baseline to compare the performance of other models. It 
# predicts the mean of the target variable for regression and the mode of the target variable 
# for classification.
#
# [input] is training data and labels
# [output] is trained model and logs
#
class NullModel:
    def __init__(self):
        self.hyperparameters = None
        self.prediction_type = None
        self.function = None
        self.positive_class = None # positive class (if binary classification is used)
        
    # Set the hyperparameters for the model (none for null model)
    def set_params(self, hyperparameters):
        self.hyperparameters = hyperparameters

    # Train the model
    def fit(self, X, y):
        # learning curve: should produce logs of model metric as function of % of trainng data 
        # used for training. There is no learning curve associated with the null model.
        learning_metrics = None
        # for classification, the function is the mode of the target variable
        if self.prediction_type == 'classification':
            self.function = y.mode()[0]
        # for regression, the function is the mean of the target variable
        elif self.prediction_type == 'regression':
            self.function = y.mean()
        return learning_metrics
    
    # Predict the target values
    def predict(self, X):
        y_pred = []
        # for each query point in X, predict the target value using the mean or mode of the target variable
        for i in range(len(X)):
            y_pred.append(self.function)
        y_pred = pd.DataFrame(y_pred) # convert to dataframe object
        return y_pred


