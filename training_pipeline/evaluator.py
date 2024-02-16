# input is predictions and labels
# output is evaluation metrics
class Evaluator:
    def __init__(self, model, config):
        self.config = config # config file for model contains what metrics to use for evaluation
        self.prediction_type = model.prediction_type # get the type of prediction
        self.positive_class = model.positive_class # get the positive class (if binary classification)
        self.metrics = None

    # Evaluate the model
    def evaluate(self, labels, predictions):
        if self.prediction_type == 'classification':
            self.classification_metrics(labels, predictions)
        elif self.prediction_type == 'regression':
            self.regression_metrics(labels, predictions)
        #print('Evaluating the model...')
        return self.metrics
    

    def count_positives_negatives(self, labels, predictions):
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        # convert all values to strings for comparison. positive class is also a string.
        for i in range(len(labels)):
            if str(predictions.iloc[i,0]) == str(labels.iloc[i]) and str(labels.iloc[i]) == self.positive_class:
                TP += 1
            elif str(predictions.iloc[i,0]) == self.positive_class and str(labels.iloc[i]) != self.positive_class:
                FP += 1
            elif str(predictions.iloc[i,0]) != self.positive_class and str(labels.iloc[i]) != self.positive_class:
                TN += 1
            elif str(predictions.iloc[i,0]) != self.positive_class and str(labels.iloc[i]) == self.positive_class:
                FN += 1
        return TP, FP, TN, FN

    # need to modify to handle multiclass classification
    def compute_confusion_matrix(self, labels, predictions):
        # if binary classification, then use positive class
        if self.positive_class != '':
            TP, FP, TN, FN = self.count_positives_negatives(labels, predictions)
            #confusion_matrix = [[TP, FP], [FN, TN]]
        # if multiclass classification, then use all classes
        else:
            # get unique classes from labels
            classes = sorted(set(labels))
            # encode classes as integers to use as indices in confusion matrix
            #class_to_index = {c: i for i, c in enumerate(classes)}
            # convert labels, predictions, and classes to integers
            #labels = [class_to_index[label] for label in labels]
            #predictions = [class_to_index[prediction] for prediction in predictions]
            #classes = [class_to_index[c] for c in classes]
            # initialize confusion matrix with zeros with size NxN where N is the number of classes
            #confusion_matrix = [[0 for _ in classes] for _ in classes]
            # initialize counters for TP, TN, FP, FN
            TP = {c: 0 for c in classes}
            TN = {c: 0 for c in classes}
            FP = {c: 0 for c in classes}
            FN = {c: 0 for c in classes}
            # calculate the confusion matrix and TP, TN, FP, FN for each class
            for label, prediction in zip(labels, predictions.iloc[:, 0].values):
                #confusion_matrix[label][prediction] += 1
                for c in classes:
                    # case where label is positive class
                    if label == c:
                        # correctly predicted positive class
                        if prediction == c:
                            TP[c] += 1
                        # incorrectly predicted as negative class
                        else:
                            FN[c] += 1
                    # case where label is negative class
                    else:
                        # correctly predicted negative class
                        if prediction == c:
                            FP[c] += 1
                        # incorrectly predicted as positive class
                        else:
                            TN[c] += 1 
            # calculate average TP, TN, FP, FN for all classes
            TP = sum(TP.values()) / len(classes)
            TN = sum(TN.values()) / len(classes)
            FP = sum(FP.values()) / len(classes)
            FN = sum(FN.values()) / len(classes)
        return TP, FP, TN, FN
    

    # calculate classification metrics
    def classification_metrics(self, labels, predictions):
        target_metrics = dict(self.config.items('evaluation_metrics'))
        TP, FP, TN, FN = self.compute_confusion_matrix(labels, predictions)
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
        ss_residual = sum([(label - prediction)**2 for label, prediction in zip(labels, predictions.iloc[:, 0].values)])
        ss_total = sum([(label - mean)**2 for label in labels])
        return 1 - (ss_residual / ss_total)
    
    def compute_mse(self, labels, predictions):
        return sum([(label - prediction)**2 for label, prediction in zip(labels, predictions.iloc[:, 0].values)]) / len(labels)

    def compute_mae(self, labels, predictions):
        return sum([abs(label - prediction) for label, prediction in zip(labels, predictions.iloc[:, 0].values)]) / len(labels)

    def compute_rmse(self, labels, predictions):
        return (sum([(label - prediction)**2 for label, prediction in zip(labels, predictions.iloc[:, 0].values)]) / len(labels))**0.5

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
    