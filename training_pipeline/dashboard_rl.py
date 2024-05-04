import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os
import imageio
import numpy as np

# [description] this class is responsible for visualizing the model's performance using learning curve and
# validation curve. It exports the dashboard as a .png file.
#
# [input] is model name, dataset name, cross validation splits, and output directory
# [output] is dashboard visuals
class Dashboard:
    def __init__(self, config):
        self.model, self.data, self.splits, self.output = self.process_config(config)

    # Process the config dictionary
    def process_config(self, config):
        # Extract model name from the model path
        model_name = os.path.basename(os.path.normpath(config.model))
        # Extract dataset name from the dataset path
        dataset_name = os.path.basename(os.path.normpath(config.data))
        # Extract number of cross validation splits
        splits = config.splits
        # Extract output directory 
        output_directory = config.output
        # Create temp directory to store images for animation
        os.mkdir(os.path.join(output_directory, 'temp'))
        return (model_name, dataset_name, splits, output_directory)

    # Visualize the model's performance
    def visualize(self, metrics, logs):
        if len(logs['learning_metrics']) != 0:
            engine = logs['learning_metrics'][0]['model'].engine
            # plots for value iteration
            if engine == 'value_iteration':
                learning_curve_fig = self.plot_learning_curve_model_based(logs['learning_metrics'])
                self.export_dashboard(learning_curve_fig, None, id=engine)
            # plots for Q-learning or SARSA
            else: 
                learning_curve_fig = self.plot_learning_curve_model_free(logs['learning_metrics'])
                self.export_dashboard(learning_curve_fig, None, id=engine)
                visits_fig = self.plot_visits_history(logs['learning_metrics'][0]['model'])
                self.export_dashboard(visits_fig, None, id= engine + '_visits')
        if len(logs['validation_metrics']) != 0:
            validation_curve_fig = self.plot_validation_curve(logs['validation_metrics'])
            self.export_dashboard(None, validation_curve_fig, id=None)
        return None

    # Plot the learning curve for Q-learning or SARSA
    def plot_learning_curve_model_free(self, learning_metrics):
        print('Plotting the learning curve...')
        model = learning_metrics[0]['model']
        Q_metrics = learning_metrics[0]['learning_metrics'][0]
        visits_metrics = learning_metrics[0]['learning_metrics'][1]
        Q_forbidden_max = Q_metrics['Q_forbidden_max']
        Q_track_max = Q_metrics['Q_track_max']
        Q_forbidden_mean = Q_metrics['Q_forbidden_mean']
        Q_track_mean = Q_metrics['Q_track_mean']
        iterations = list(range(len(Q_track_max)))
        fig, ax = plt.subplots(3, 2, squeeze=False, dpi = 300)
        fig.suptitle('Learning Curves for ' + model.engine.capitalize(), fontsize=8)
        ax[0][0].plot(iterations, Q_forbidden_max, color='blue')
        ax[0][0].set_title('Max of Q Values for Forbidden States Over Iterations', fontsize=6)
        ax[0][0].set_xlabel('Iterations', fontsize=6)
        ax[0][0].set_ylabel('Q Forbidden Max', fontsize=6)
        ax[0][0].tick_params(axis='both', which='major', labelsize=6)
        ax[0][0].grid(True)
        ax[1][0].plot(iterations, Q_forbidden_mean, color='blue')
        ax[1][0].set_title('Mean of Q Values for Forbidden States Over Iterations', fontsize=6)
        ax[1][0].set_xlabel('Iterations', fontsize=6)
        ax[1][0].set_ylabel('Q Forbidden Mean', fontsize=6)
        ax[1][0].tick_params(axis='both', which='major', labelsize=6)
        ax[1][0].grid(True)
        ax[0][1].plot(iterations, Q_track_max, color='orange')
        ax[0][1].set_title('Max of Q Values for Valid States Over Iterations', fontsize=6)
        ax[0][1].set_xlabel('Iterations', fontsize=6)
        ax[0][1].set_ylabel('Q Valid Max', fontsize=6)
        ax[0][1].tick_params(axis='both', which='major', labelsize=6)
        ax[0][1].grid(True)
        ax[1][1].plot(iterations, Q_track_mean, color='orange')
        ax[1][1].set_title('Mean of Q Values for Valid States Over Iterations', fontsize=6)
        ax[1][1].set_xlabel('Iterations', fontsize=6)
        ax[1][1].set_ylabel('Q Valid Mean', fontsize=6)
        ax[1][1].tick_params(axis='both', which='major', labelsize=6)
        ax[1][1].grid(True)
        ax[2][0].plot(iterations, visits_metrics, color='orange')
        ax[2][0].set_title('Percentage of Unvisited States Over Iterations', fontsize=6)
        ax[2][0].set_xlabel('Iterations', fontsize=6)
        ax[2][0].set_ylabel('Unvisited States', fontsize=6)
        ax[2][0].tick_params(axis='both', which='major', labelsize=6)
        ax[2][0].grid(True)
        plt.tight_layout()
        return fig
    
    # Plot the learning curve for value iteration
    def plot_learning_curve_model_based(self, learning_metrics):
        fig, ax = plt.subplots(figsize=(12, 5), dpi = 300)
        print('Plotting the learning curve...')
        model = learning_metrics[0]['model']
        V_metrics = learning_metrics[0]['learning_metrics']
        iterations = list(range(len(V_metrics)))
        fig.suptitle('Learning Curves for ' + model.engine.capitalize(), fontsize=12)
        ax.plot(iterations, V_metrics, color='blue')
        ax.set_yscale('log')
        ax.set_title('Delta Change in Value Function Over Iterations', fontsize=10)
        ax.set_xlabel('Iterations', fontsize=10)
        ax.set_ylabel('Delta V', fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.grid(True)
        return fig

    def plot_visits_history(self, model):
        print('Plotting the visits history...')
        visits = model.visit_history
        world = model.world
        heat_map = np.zeros((len(world), len(world[0])))
        fig, ax = plt.subplots(figsize=(12, 5), dpi = 300, constrained_layout=True)
        # create heatmap of counts of visits to each state 
        visit_count = [sum(state) for state in visits]
        # convert state index to states in form (x, y)
        for s in range(len(visit_count)):
            state = model.index_to_state[s]
            heat_map[state[1]][state[0]] += visit_count[s]
        # extract track states from heat map
        track_heat_map = [heat_map[i][j] for i in range(len(world)) for j in range(len(world[i])) if world[i][j] != model.forbidden_state]
        # calculate stats of heat map 
        mean = np.mean(track_heat_map)
        std = np.std(track_heat_map)
        # plot stats in title 
        ax.set_title('Visits Frequency\nMean: ' + str(round(mean, 3)) + ' Std: ' + str(round(std, 3)), fontsize=14)
        plt.imshow(heat_map, cmap='hot', interpolation='nearest')
        plt.colorbar()
        return fig
    
    # Plot the validation curve
    def plot_validation_curve(self, validation_metrics):
        print('Plotting the validation curve...')
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(12, 5), dpi = 300, constrained_layout=True)
        plt.grid()
        ax.set_title('Validation Curve', fontsize=10)
        ax.set_xlabel('Hyperparameters')
        ax.set_ylabel('Model Metric(s)')
        x = len(validation_metrics[0])
        x_labels = []
        y = {}
        # Extract the hyperparameters for each model
        for model in validation_metrics[0]:
            hyperparameters, _ = model
            formatted_hyperparameters = '\n'.join(str(hyperparameters).split(', '))
            formatted_hyperparameters = formatted_hyperparameters.replace('{', '').replace('}', '').replace('hidden_layer_1_size', 'hidden_layer_1').replace('hidden_layer_2_size', 'hidden_layer_2')
            x_labels.append(formatted_hyperparameters)
        # Extract the metrics for each model
        for metric in validation_metrics[0][0][1].items():
            y[metric[0]] = []
            for model in validation_metrics[0]:
                _, metrics = model
                y[metric[0]].append(metrics[metric[0]])
        # Plot the hyperparameters and metrics
        for metric in y.items():
            ax.bar(range(x), metric[1], width=0.2, label = 'testing ' + metric[0], zorder=3)
            # ax.scatter(range(x), metric[1])
            # ax.plot(range(x), metric[1], label = 'testing ' + metric[0])
        y_min = min(min(metric) for metric in y.values())
        y_max = max(max(metric) for metric in y.values())
        ax.set_ylim(y_min*0.975, y_max*1.01)
        ax.legend()
        plt.xticks(range(x), x_labels, rotation=0, fontsize=8)
        fig.subplots_adjust(bottom=0.5)
        # plt.tight_layout()
        return fig

    # Export the dashboard as a .png file
    def export_dashboard(self, learning_curve_fig, validation_curve_fig, id=None):
        print('Exporting the dashboard...')
        if learning_curve_fig is not None:
            if id is None: id = ''
            learning_curve_name = self.model + ' learning curves for ' + self.data + ' using ' + str(id) + '.png'
            learning_curve_fullpath = os.path.join(self.output, learning_curve_name)
            learning_curve_fig.savefig(learning_curve_fullpath)
        if validation_curve_fig is not None:
            validation_curve_name = self.model + ' validation curves for ' + self.data + ' using ' + str(id) + '.png'
            validation_curve_fullpath = os.path.join(self.output, validation_curve_name)
            validation_curve_fig.savefig(validation_curve_fullpath)      
        return None