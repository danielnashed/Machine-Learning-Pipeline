import matplotlib.pyplot as plt
import os

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
        return (model_name, dataset_name, splits, output_directory)

    # Visualize the model's performance
    def visualize(self, metrics, logs):
        if len(logs['learning_metrics']) != 0:
            learning_curve_fig = self.plot_learning_curve(logs['learning_metrics'])
            self.export_dashboard(learning_curve_fig, None)
        if len(logs['validation_metrics']) != 0:
            validation_curve_fig = self.plot_validation_curve(logs['validation_metrics'])
            self.export_dashboard(None, validation_curve_fig)
        return None
    
    # Plot the learning curve (TO BE IMPLEMENTED LATER IN PROJECT 2)   
    def plot_learning_curve(self, learning_metrics):
        print('Plotting the learning curve...')
        # Create a figure and axis
        fig, ax = plt.subplots(dpi = 300)
        ax.set_title('Learning Curve (TO BE IMPLEMENTED LATER IN PROJECT 2)', fontsize=10)
        ax.set_xlabel('Percent of Training Data Used')
        ax.set_ylabel('Model Metric(s)')
        plt.grid()
        plt.tight_layout()
        return fig
    
    # Plot the validation curve
    def plot_validation_curve(self, validation_metrics):
        print('Plotting the validation curve...')
        # Create a figure and axis
        fig, ax = plt.subplots(dpi = 300)
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
            x_labels.append(str(hyperparameters))
        # Extract the metrics for each model
        for metric in validation_metrics[0][0][1].items():
            y[metric[0]] = []
            for model in validation_metrics[0]:
                _, metrics = model
                y[metric[0]].append(metrics[metric[0]])
        # Plot the hyperparameters and metrics
        for metric in y.items():
            ax.scatter(range(x), metric[1])
            ax.plot(range(x), metric[1], label = 'testing ' + metric[0])
        ax.legend()
        plt.xticks(range(x), x_labels, rotation=90)
        plt.tight_layout()
        return fig

    # Export the dashboard as a .png file
    def export_dashboard(self, learning_curve_fig, validation_curve_fig):
        print('Exporting the dashboard...')
        if learning_curve_fig is not None:
            learning_curve_name = self.model + ' learning curve with ' + self.data + 'dataset using k_x_' + str(self.splits) + ' cross validation.png'
            learning_curve_fullpath = os.path.join(self.output, learning_curve_name)
            learning_curve_fig.savefig(learning_curve_fullpath)
        if validation_curve_fig is not None:
            validation_curve_name = self.model + ' validation curve with ' + self.data + 'dataset using k_x_' + str(self.splits) + ' cross validation.png'
            validation_curve_fullpath = os.path.join(self.output, validation_curve_name)
            validation_curve_fig.savefig(validation_curve_fullpath)      
        return None