import copy
import os
from itertools import product
import pickle
import time
import configparser
import math
import numpy as np
from inference_pipeline.reinforcement_learning import path_visualizer as __path_visualizer__

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
        self.num_exps =  int(dict(self.config.items('model'))['num_exps']) # Number of training iterations
    
    # Get the favorite metric for model selection
    def get_favorite_metric(self):
        metrics = self.config.items('metric_for_model_selection')
        for key, value in metrics:
            if value == '1':
                favorite_metric = key
                break
        return favorite_metric
    
    # Get the best model based on the favorite metric
    def get_best_model(self, models):
        best_model = min(models, key=lambda x: x[self.favorite_metric])
        return best_model
    
    # Get the average of the metrics for all experiments
    def get_average(self, metrics_all_experiements):
        metrics_averaged = {}
        # calculate average of each metric
        for metric in metrics_all_experiements[0].keys():
            metrics_averaged[metric] = sum([experiment[metric] for experiment in metrics_all_experiements if not math.isnan(experiment[metric])]) / len(metrics_all_experiements)
        return metrics_averaged
    
    # Get the standard deviation of the metrics for all experiments
    def get_std(self, metrics_all_experiements, metrics_average):
        metrics_std = {}
        # calculate standard deviation of each metric
        for metric in metrics_all_experiements[0].keys():
            metrics_std[metric] = math.sqrt(sum([(experiment[metric] - metrics_average[metric])**2 for experiment in metrics_all_experiements if not math.isnan(experiment[metric])]) / len(metrics_all_experiements))
        return metrics_std
    
    def check_over_under_flow(self, x):
        for key in x:
            if np.isnan(x[key]) or np.isinf(x[key]):
                x[key] = np.nan_to_num(x[key])
        return x
    
    def get_stats(self, metrics_all_experiements):
        metrics_all_experiements = [self.check_over_under_flow(metrics) for metrics in metrics_all_experiements]
        metrics_average = self.get_average(metrics_all_experiements)
        metrics_std = self.get_std(metrics_all_experiements, metrics_average)
        # round the metrics to 3 decimal places
        metrics_average = {key + '_mean': round(value, 3) for key, value in metrics_average.items()}
        metrics_std = {key + '_std': round(value, 3) for key, value in metrics_std.items()}
        return (metrics_average, metrics_std)
    

    # Train the model
    def train_model(self, data):
        start_time = time.time()
        print('\nTraining the model...')
        metrics_all_experiements = []
        models = []
        status_all = []
        dist_to_goal_all = []
        self.collisions_all = []
        self.paths = []
        # create a deep copy of the model to save it
        model = copy.deepcopy(self.model)
        model.output = self.output # path to save model functions to
        learning_metrics = model.fit(data) # learn optimal policy
        # learning curve: should produce logs of model metric as function of training data size
        self.save_logs(validation_metrics = None, learning_metrics = {'model': model, 'learning_metrics': learning_metrics})
        models.append(model) # save the model
        # for each experiment, apply the policy and evaluate the path taken by agent
        for i in range(self.num_exps):
            print(f"    Experiment: {i + 1} of {self.num_exps}")
            path, path_metrics, status, dist_to_goal, collisions = model.predict() # apply optimal policy to drive agent in world
            # evaluate the model and save the metrics
            metrics_all_experiements.append(path_metrics)
            self.paths.append(path) # save the path
            status_all.append(status)
            dist_to_goal_all.append(dist_to_goal)
            self.collisions_all.append(collisions)
        # average the metrics for all experiments
        metrics_average, metrics_std = self.get_stats(metrics_all_experiements)
        # number of successful paths that reach goal 
        success_rate = len([status for status in status_all if status is True]) * 100 / self.num_exps
        # paths that reach goal
        successful_paths = [len(path) for status, path in zip(status_all, self.paths) if status == True]
        if len(successful_paths) == 0:
            average_steps = 0
        else:
            average_steps = sum(successful_paths) / len(successful_paths)
        # average collisions of agent 
        average_collisions = sum(self.collisions_all) / len(self.collisions_all)
        # average speed of agent 
        average_speed = 0
        for status, path in zip(status_all, self.paths):
            if status == True:
                velocity = np.array([0, 0])
                for triplet in path:
                    velocity += np.array(triplet[0][2:]) # current velocity in form [vx, vy]
                average_velocity = velocity / len(path)
                average_speed += np.linalg.norm(average_velocity)
        if len(successful_paths) == 0:
            average_speed = 0
        else:
            average_speed = average_speed / len(successful_paths)
        # save all average metrics over all paths 
        self.average_metrics = {'success_rate': success_rate, 
                                'average_steps': average_steps, 
                                'average_speed': average_speed,
                                'average_collisions': average_collisions}
        # select the best model based on the favorite metric
        self.model = models[0]
        end_time = time.time()
        print(f"\nMetrics for trained model: {metrics_average} {metrics_std} --- Success rate: {success_rate:.2f}% --- Average steps to get to goal: {average_steps:.2f} --- Average collisions: {average_collisions:.2f} --- Training time: {end_time - start_time:.2f}s")
        return None

    # # Train the model
    # def train_model(self, data):
    #     start_time = time.time()
    #     print('\nTraining the model...')
    #     metrics_all_experiements = []
    #     models = []
    #     # for each experiment, train the model and evaluate it
    #     for i in range(10):
    #         print(f"    Experiment: {i + 1} of {10}")
    #         # create a deep copy of the model to save it
    #         model = copy.deepcopy(self.model)
    #         learning_metrics = model.fit(data) # learn optimal policy
    #         # learning curve: should produce logs of model metric as function of training data size
    #         self.save_logs(validation_metrics = None, learning_metrics = {'model': model, 'learning_metrics': learning_metrics})
    #         path, path_metrics = model.predict() # apply optimal policy to drive agent in world
    #         # evaluate the model and save the metrics
    #         metrics_all_experiements.append(path_metrics)
    #         models.append(model) # save the model
    #     # average the metrics for all experiments
    #     metrics_average, metrics_std = self.get_stats(metrics_all_experiements)
    #     # select the best model based on the favorite metric
    #     best_model = models[metrics_all_experiements.index(self.get_best_model(metrics_all_experiements))]
    #     self.model = best_model
    #     end_time = time.time()
    #     print(f"\nMetrics for trained model: {metrics_average} {metrics_std} --- Training time: {end_time - start_time:.2f}s")
    #     return None
    
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
        model_name = self.model.__class__.__name__
        full_path = os.path.join(self.output, model_name + '.pickle')
        with open(full_path, 'wb') as f:
            pickle.dump(self.model, f)
        # export the path taken by agent for all 10 experiments
        directory_image_path = os.path.join(self.output, 'paths')
        os.makedirs(directory_image_path, exist_ok=True)
        for i, path in enumerate(self.paths):
            collisions = self.collisions_all[i]
            image_path = os.path.join(directory_image_path, f'path_{i+1}.txt')
            path_visualizer = __path_visualizer__.PathVisualizer(self.model, image_path, collisions, self.average_metrics)
            path_visualizer.visualize_path(path)
        return None

    # Run the learner
    def learn(self, data):
        self.train_model(data)
        self.export_model()
        self.export_logs()
        return (self.model, self.logs)