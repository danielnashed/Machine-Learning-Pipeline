import configparser
import copy
import os
from itertools import product
import pickle
import time
from training_pipeline import evaluator as __evaluator__

# input is processed data and model
# output is model
class Learner:
    def __init__(self, model_meta, output):
        self.model, self.config = model_meta
        self.output = output
        self.logs = {'validation_metrics': [], 'learning_metrics': []}
        self.favorite_metric = self.get_favorite_metric()
    
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
        metrics_all_models = []
        parameters = [line[0] for line in hyperparameters] # extract names of hyperparameters
        values = [line[1] for line in hyperparameters] # extract values of each hyperparameter
        hyperparameters = list(product(*[eval(parameter) for parameter in values])) # find all possible combinations of hyperparameters
        for i, combination in enumerate(hyperparameters):
            print(f"Finetuning model hyperparameters: {dict(zip(parameters, combination))} --- {i*100/len(hyperparameters):.2f}% complete")
            model = self.model
            model.set_params(dict(zip(parameters, combination)))
            evaluator = __evaluator__.Evaluator(self.model, self.config)
            metrics_all_experiements = []
            for experiment in data:
                print(f"    Experiment: {len(metrics_all_experiements) + 1} of {len(data)}")
                X_train, y_train, X_validation, y_validation = experiment
                model.fit(X_train, y_train)
                y_validation_pred = model.predict(X_validation)
                metrics_all_experiements.append(evaluator.evaluate(y_validation, y_validation_pred))
            metrics_averaged = self.get_average(metrics_all_experiements)
            metrics_all_models.append((dict(zip(parameters, combination)), metrics_averaged))
        best_model = max(metrics_all_models, key=lambda x: x[1][self.favorite_metric])
        self.model.set_params(best_model[0])
        # validation curve: should produce logs of model metric as function of hyperparameter
        self.save_logs(validation_metrics = metrics_all_models, learning_metrics = None)
        end_time = time.time()
        print(f"\nBest hyperparameters: {best_model[0]} --- Hypertuning time: {end_time - start_time:.2f}s")

    def get_average(self, metrics_all_experiements):
        metrics_averaged = {}
        for metric in metrics_all_experiements[0].keys():
            metrics_averaged[metric] = sum([experiment[metric] for experiment in metrics_all_experiements]) / len(metrics_all_experiements)
        return metrics_averaged
    
    # Train the model
    def train_model(self, data):
        start_time = time.time()
        print('\nTraining the model...')
        evaluator = __evaluator__.Evaluator(self.model, self.config)
        metrics_all_experiements = []
        models = []
        for experiment in data:
            print(f"    Experiment: {len(metrics_all_experiements) + 1} of {len(data)}")
            X_train, y_train, X_test, y_test = experiment
            learning_metrics = self.model.fit(X_train, y_train)
            model = copy.deepcopy(self.model)
            self.save_logs(validation_metrics = None, learning_metrics = {'model': model, 'learning_metrics': learning_metrics})
            models.append(model) # save the model
            y_test_pred = self.model.predict(X_test)
            metrics_all_experiements.append(evaluator.evaluate(y_test, y_test_pred))
        metrics_averaged = self.get_average(metrics_all_experiements)
        best_model = models[metrics_all_experiements.index(max(metrics_all_experiements, key=lambda x: x[self.favorite_metric]))]
        self.model = best_model
        end_time = time.time()
        print(f"\nAverage metrics for trained model: {metrics_averaged} --- Training time: {end_time - start_time:.2f}s")

    def save_logs(self, validation_metrics: None, learning_metrics: None):
        # Save the logs
        if validation_metrics:
            self.logs['validation_metrics'].append(validation_metrics)
        if learning_metrics:
            self.logs['learning_metrics'].append(learning_metrics)
    
    # Export the logs
    def export_logs(self):
        with open(os.path.join(self.output, 'logs.pickle'), 'wb') as f:
            pickle.dump(self.logs, f)
        print('Exporting the logs...')

    def export_model(self):
        # Export the model
        full_path = os.path.join(self.output, self.model.__class__.__name__ + '.pickle')
        with open(full_path, 'wb') as f:
            pickle.dump(self.model, f)
        print('Exporting the model...')

    # Run the learner
    def learn(self, data):
        train_validation_data, train_test_data = data
        self.hyperparameter_tuning(train_validation_data)
        self.train_model(train_test_data)
        self.export_model()
        self.export_logs()
        return (self.model, self.logs)