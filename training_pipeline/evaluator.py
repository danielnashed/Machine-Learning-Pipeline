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
    

    # need to modify to handle multiclass classification
    def compute_confusion_matrix(self, labels, predictions):
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for i in range(len(labels)):
            if predictions.iloc[i,0] == labels.iloc[i]:
                TP += 1
            # if labels[i] == 1 and predictions[i] == 1:
            #     TP += 1
            # elif labels[i] == 0 and predictions[i] == 1:
            #     FP += 1
            # elif labels[i] == 0 and predictions[i] == 0:
            #     TN += 1
            # elif labels[i] == 1 and predictions[i] == 0:
            #     FN += 1
        return TP, FP, TN, FN
    

    # calculate classification metrics
    def classification_metrics(self, labels, predictions):
        TP, FP, TN, FN = self.compute_confusion_matrix(labels, predictions)
        target_metrics = dict(self.config.items('evaluation_metrics'))
        metrics = {}
        if int(target_metrics['accuracy']) == 1:
            accuracy = (TP + TN) / len(predictions)
            metrics['accuracy'] = accuracy
        if int(target_metrics['precision']) == 1:
            precision = TP / (TP + FP)
            metrics['precision'] = precision
        if int(target_metrics['recall']) == 1:
            recall = TP / (TP + FN)
            metrics['recall'] = recall
        if int(target_metrics['f1']) == 1:
            f1 = 2 * (precision * recall) / (precision + recall)
            metrics['f1'] = f1
        self.metrics = metrics
    
    def compute_r2(self, labels, predictions):
        mean = sum(labels) / len(labels)
        ss_residual = sum([(label - prediction)**2 for label, prediction in zip(labels, predictions)])
        ss_total = sum([(label - mean)**2 for label in labels])
        return 1 - (ss_residual / ss_total)
    
    def compute_mse(self, labels, predictions):
        return sum([(label - prediction)**2 for label, prediction in zip(labels, predictions)]) / len(labels)

    def compute_mae(self, labels, predictions):
        return sum([abs(label - prediction) for label, prediction in zip(labels, predictions)]) / len(labels)

    def compute_rmse(self, labels, predictions):
        return (sum([(label - prediction)**2 for label, prediction in zip(labels, predictions)]) / len(labels))**0.5

    # calculate regression metrics
    def regression_metrics(self, labels, predictions):
        target_metrics = dict(self.config.items('evaluation_metrics'))
        metrics = {}
        if int(target_metrics['r2']) == 1:
            r2 = self.compute_r2(labels, predictions)
            metrics['r2'] = r2
        if int(target_metrics['mse']) == 1:
            mse = self.compute_mse(labels, predictions)
            metrics['mse'] = mse
        if int(target_metrics['mae']) == 1:
            mae = self.compute_mae(labels, predictions)
            metrics['mae'] = mae
        if int(target_metrics['rmse']) == 1:
            rmse = self.compute_rmse(labels, predictions)
            metrics['rmse'] = rmse
        self.metrics = metrics
    