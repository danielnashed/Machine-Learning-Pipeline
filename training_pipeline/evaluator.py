# input is predictions and labels
# output is evaluation metrics
class Evaluator:
    def __init__(self, model, config):
        self.config = config # config file for model contains what metrics to use for evaluation
        self.prediction_type = model.prediction_type # get the type of prediction
        self.metrics = None

    # Evaluate the model
    def evaluate(self, labels, predictions):
        if self.prediction_type == 'classification':
            self.classification_metrics(labels, predictions)
        elif self.prediction_type == 'regression':
            self.regression_metrics(labels, predictions)
        print('Evaluating the model...')
        return self.metrics
    
    # calculate classification metrics
    def classification_metrics(self, labels, predictions):
        target_metrics = dict(self.config.items('evaluation_metrics'))
        metrics = {}
        if int(target_metrics['accuracy']) == 1:
            accuracy = 0
            metrics['accuracy'] = accuracy
        if int(target_metrics['precision']) == 1:
            precision = 0
            metrics['precision'] = precision
        if int(target_metrics['recall']) == 1:
            recall = 0
            metrics['recall'] = recall
        if int(target_metrics['f1']) == 1:
            f1 = 0
            metrics['f1'] = f1
        self.metrics = metrics
    
    # calculate regression metrics
    def regression_metrics(self, labels, predictions):
        target_metrics = dict(self.config.items('evaluation_metrics'))
        metrics = {}
        if int(target_metrics['r2']) == 1:
            r2 = 0
            metrics['r2'] = r2
        if int(target_metrics['mse']) == 1:
            mse = 0
            metrics['mse'] = mse
        if int(target_metrics['mae']) == 1:
            mae = 0
            metrics['mae'] = mae
        if int(target_metrics['rmse']) == 1:
            rmse = 0
            metrics['rmse'] = rmse
        self.metrics = metrics
    