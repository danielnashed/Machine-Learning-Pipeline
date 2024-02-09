import configparser
import copy
from itertools import product
from training_pipeline import evaluator as __evaluator__

# input is processed data and model
# output is model
class Learner:
    def __init__(self, model_meta):
        self.model, self.config = model_meta
        self.logs = {}

    # Hyperparameter tuning
    def hyperparameter_tuning(self, data):
        hyperparameters = self.config.items('hyperparameters')
        metrics_all_models = []
        parameters = [line[0] for line in hyperparameters] # extract names of hyperparameters
        values = [line[1] for line in hyperparameters] # extract values of each hyperparameter
        hyperparameters = list(product(*[eval(parameter) for parameter in values])) # find all possible combinations of hyperparameters
        for combination in hyperparameters:
            model = self.model
            model.set_params(dict(zip(parameters, combination)))
            evaluator = __evaluator__.Evaluator(self.model, self.config)
            metrics_all_experiements = []
            for experiment in data:
                X_train, y_train, X_validation, y_validation = experiment
                model.fit(X_train, y_train)
                y_validation_pred = model.predict(X_validation)
                metrics_all_experiements.append(evaluator.evaluate(y_validation, y_validation_pred))
            metrics_averaged = self.get_average(metrics_all_experiements)
            metrics_all_models.append((dict(zip(parameters, combination)), metrics_averaged))
        best_model = max(metrics_all_models, key=lambda x: x[1]['accuracy'])
        self.model.set_params(best_model[0])
        # validation curve: should produce logs of model metric as function of hyperparameter
        self.export_logs(validation_metrics = metrics_all_models, learning_metrics = None)
        print('Finetuning model hyperparameters...')
        print(f"Best hyperparameters: {best_model[0]}")
        return None

    def get_average(self, metrics_all_experiements):
        metrics_averaged = {}
        for metric in metrics_all_experiements[0].keys():
            metrics_averaged[metric] = sum([experiment[metric] for experiment in metrics_all_experiements]) / len(metrics_all_experiements)
        return metrics_averaged
    
    # Train the model
    def train_model(self, data):
        evaluator = __evaluator__.Evaluator(self.model, self.config)
        metrics_all_experiements = []
        models = []
        for experiment in data:
            X_train, y_train, X_test, y_test = experiment
            learning_metrics = self.model.fit(X_train, y_train)
            model = copy.deepcopy(self.model)
            self.export_logs(validation_metrics = None, learning_metrics = {'model': model, 'learning_metrics': learning_metrics})
            models.append(model) # save the model
            y_test_pred = self.model.predict(X_test)
            metrics_all_experiements.append(evaluator.evaluate(y_test, y_test_pred))
        metrics_averaged = self.get_average(metrics_all_experiements)
        best_model = models[metrics_all_experiements.index(max(metrics_all_experiements, key=lambda x: x['accuracy']))]
        self.model = best_model
        print('Training the model...')
        print(f"Average metrics for trained model: {metrics_averaged: .3f}")
        return None

    def export_model(self):
        # Export the model
        print('Exporting the model...')
        return None

    def export_logs(self, validation_metrics: None, learning_metrics: None):
        # Export the logs
        if validation_metrics:
            self.logs['validation_metrics'] = validation_metrics
        if learning_metrics:
            self.logs['learning_metrics'] = learning_metrics
        print('Exporting the logs...')
        return None

    # Run the learner
    def learn(self, data):
        train_validation_data, train_test_data = data
        self.hyperparameter_tuning(train_validation_data)
        self.train_model(train_test_data)
        return (self.model, self.logs)