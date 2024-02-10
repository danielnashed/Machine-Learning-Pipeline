import pandas as pd
class NullModel:
    def __init__(self):
        self.rubbish = ''
        self.hyperparameters = None
        self.prediction_type = None
        self.function = None
        self.positive_class = None # positive class (if binary classification is used)
    
    # def set_params(self, **kwargs):
    #     pass
        
    def set_params(self, hyperparameters):
        self.hyperparameters = hyperparameters
        # self.hyperparameters = {"gamma": hyperparameters[0],
        #                         "C": hyperparameters[1], 
        #                         "kernel": hyperparameters[2]}

    def fit(self, X, y):
        # learning curve: should produce logs of model metric as function of % of trainng data used for training
        learning_metrics = None
        if self.prediction_type == 'classification':
            self.function = y.mode()[0]
        elif self.prediction_type == 'regression':
            self.function = y.mean()
        return learning_metrics
    
    def predict(self, X):
        y_pred = []
        for i in range(len(X)):
            y_pred.append(self.function)
        y_pred = pd.DataFrame(y_pred) # convert to dataframe object
        return y_pred


