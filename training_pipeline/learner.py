import copy
import os
from itertools import product
import pickle
import time
import configparser
import math
from training_pipeline import evaluator as __evaluator__

# [description] this class is responsible for training the model and tuning its hyperparameters. It
# also exports the model and logs for the model.
#
# [input] is untrained model and data
# [output] is trained model and logs
#
class Learner:
    def __init__(self, model_meta, output):
        self.model, self.config = model_meta # Extract the instantiated model class and its config file
        self.output = output # Output directory
        self.logs = {'validation_metrics': [], 'learning_metrics': []}
        self.favorite_metric = self.get_favorite_metric() # Get the favorite metric for model selection
    
    # Get the favorite metric for model selection
    def get_favorite_metric(self):
        metrics = self.config.items('metric_for_model_selection')
        for key, value in metrics:
            if value == '1':
                favorite_metric = key
                break
        return favorite_metric

    # Hyperparameter tuning
    def hyperparameter_tuning(self, data):
        start_time = time.time()
        hyperparameters = self.config.items('hyperparameters')
        if len(hyperparameters) == 0:
            print('No hyperparameters to tune.')
            return None
        metrics_all_models = []
        parameters = [line[0] for line in hyperparameters] # extract names of hyperparameters
        values = [line[1] for line in hyperparameters] # extract values of each hyperparameter
        # find all possible combinations of hyperparameters
        hyperparameters = list(product(*[eval(parameter) for parameter in values]))
        # for each combination of hyperparameters, train the model and evaluate it
        for i, combination in enumerate(hyperparameters):
            print(f"Finetuning model hyperparameters: {dict(zip(parameters, combination))} --- {i*100/len(hyperparameters):.2f}% complete")
            model = self.model
            # set the hyperparameters for the model
            model.set_params(dict(zip(parameters, combination)))
            # initialize the evaluator class
            evaluator = __evaluator__.Evaluator(self.model, self.config)
            metrics_all_experiements = []
            # for each experiment, train the model and evaluate it
            for experiment in data:
                print(f"    Experiment: {len(metrics_all_experiements) + 1} of {len(data)}")
                X_train, y_train, X_validation, y_validation = experiment
                model.fit(X_train, y_train)
                y_validation_pred = model.predict(X_validation)
                # evaluate the model and save the metrics
                metrics_all_experiements.append(evaluator.evaluate(y_validation, y_validation_pred))
            # average the metrics for all experiments
            metrics_averaged = self.get_average(metrics_all_experiements)
            # save the hyperparameters and average metrics for the model
            metrics_all_models.append((dict(zip(parameters, combination)), metrics_averaged))
        # select the best model based on the favorite metric
        best_model = self.get_best_model(metrics_all_models)
        # set the model to the best model
        self.model.set_params(best_model[0])
        # save the logs if there are hyperparameters that were tuned
        if len(hyperparameters) != 0:
            # validation curve: should produce logs of model metric as function of hyperparameter
            self.save_logs(validation_metrics = metrics_all_models, learning_metrics = None)
        end_time = time.time()
        print(f"\nBest hyperparameters: {best_model[0]} --- Hypertuning time: {end_time - start_time:.2f}s")
        return None
    
    # Get the best model based on the favorite metric
    def get_best_model(self, models):
        prediction_type = self.model.prediction_type
        # for classification, select the model with the highest favorite metric
        if prediction_type == 'classification':
            # for hyperparameter tuning, the metrics are stored in a list of length 2
            if len(models[0]) == 2:
                best_model = max(models, key=lambda x: x[1][self.favorite_metric])
            # for training the model, the metrics are stored in a dictionary
            else:
                best_model = max(models, key=lambda x: x[self.favorite_metric])
        # for regression, select the model with the lowest favorite metric
        elif prediction_type == 'regression':
            # for hyperparameter tuning, the metrics are stored in a list of length 2
            if len(models[0]) == 2:
                best_model = min(models, key=lambda x: x[1][self.favorite_metric])
            # for training the model, the metrics are stored in a dictionary
            else:
                best_model = min(models, key=lambda x: x[self.favorite_metric])
        return best_model
    
    # Get the average of the metrics for all experiments
    def get_average(self, metrics_all_experiements):
        metrics_averaged = {}
        # calculate average of each metric
        for metric in metrics_all_experiements[0].keys():
            metrics_averaged[metric] = sum([experiment[metric] for experiment in metrics_all_experiements if not math.isnan(experiment[metric])]) / len(metrics_all_experiements)
        return metrics_averaged
    
    # Train the model
    def train_model(self, data):
        start_time = time.time()
        print('\nTraining the model...')
        evaluator = __evaluator__.Evaluator(self.model, self.config)
        metrics_all_experiements = []
        models = []
        # for each experiment, train the model and evaluate it
        for experiment in data:
            print(f"    Experiment: {len(metrics_all_experiements) + 1} of {len(data)}")
            X_train, y_train, X_test, y_test = experiment
            # create a deep copy of the model to save it
            model = copy.deepcopy(self.model)
            learning_metrics = model.fit(X_train, y_train)
            # learning curve: should produce logs of model metric as function of training data size
            self.save_logs(validation_metrics = None, learning_metrics = {'model': model, 'learning_metrics': learning_metrics})
            y_test_pred = model.predict(X_test)
            # evaluate the model and save the metrics
            metrics_all_experiements.append(evaluator.evaluate(y_test, y_test_pred))
            models.append(model) # save the model
        # average the metrics for all experiments
        metrics_averaged = self.get_average(metrics_all_experiements)
        # select the best model based on the favorite metric
        best_model = models[metrics_all_experiements.index(self.get_best_model(metrics_all_experiements))]
        self.model = best_model
        end_time = time.time()
        print(f"\nAverage metrics for trained model: {metrics_averaged} --- Training time: {end_time - start_time:.2f}s")
        return None
    
    # Save the logs
    def save_logs(self, validation_metrics: None, learning_metrics: None):
        if validation_metrics:
            self.logs['validation_metrics'].append(validation_metrics)
        if learning_metrics:
            self.logs['learning_metrics'].append(learning_metrics)
        return None
    
    # Export the logs for the model as a pickle file
    def export_logs(self):
        print('Exporting the logs...')
        with open(os.path.join(self.output, 'logs.pickle'), 'wb') as f:
            pickle.dump(self.logs, f)
        return None

    # Export the trained model as a pickle file
    def export_model(self):
        print('Exporting the model...')
        full_path = os.path.join(self.output, self.model.__class__.__name__ + '.pickle')
        with open(full_path, 'wb') as f:
            pickle.dump(self.model, f)
        return None

    # Run the learner
    def learn(self, data):
        train_validation_data, train_test_data = data
        self.hyperparameter_tuning(train_validation_data)
        self.train_model(train_test_data)
        self.export_model()
        self.export_logs()
        return (self.model, self.logs)