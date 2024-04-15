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
            learning_curve_figs = self.plot_learning_curve(logs['learning_metrics'])
            weights_biases_figs = self.plot_weights_biases(logs['learning_metrics'])
            for id, learning_curve_fig in enumerate(learning_curve_figs):
                self.export_dashboard(learning_curve_fig, None, id=id+1)
        if len(logs['validation_metrics']) != 0:
            validation_curve_fig = self.plot_validation_curve(logs['validation_metrics'])
            self.export_dashboard(None, validation_curve_fig, id=None)
        return None

    # Plot the learning curve
    def plot_learning_curve(self, learning_metrics):
        print('Plotting the learning curve...')
        figures = []
        for k, model in enumerate(learning_metrics):
            metrics = model['learning_metrics']
            if len(metrics) == 1:
                arch_name = 'Feedforward Neural Network'
                fig, ax = plt.subplots(1, 2, squeeze=False, dpi = 300)
            else:
                arch_name = 'Autoencoder + Feedforward Neural Network'
                fig, ax = plt.subplots(2, 2, squeeze=False, dpi = 300)
            fig.suptitle('Learning Curves for ' + arch_name + ' - Experiment ' + str(k+1), fontsize=8)
            for i, arch in enumerate(metrics):
                for j, metric in enumerate(arch.items()):
                    metric_name = metric[0]
                    x = list(metric[1].keys())
                    if j == 0:
                        y = list(metric[1].values())
                        ax[i][j].plot(x, y, color='red')
                    elif j == 1:
                        train_metric = [list(metric[1].values())[i][0] for i in range(len(x))]
                        val_metric = [list(metric[1].values())[i][1] for i in range(len(x))]
                        ax[i][j].plot(x, train_metric, color='blue', label='Training')
                        ax[i][j].plot(x, val_metric, color='orange', label='Validation')
                        ax[i][j].legend(fontsize=6)
                    else:
                        continue
                    ax[i][j].set_title(metric_name + ' over Training Epochs' + ' on ' + self.data.capitalize() + ' Dataset', fontsize=6)
                    ax[i][j].set_xlabel('Epoch', fontsize=6)
                    ax[i][j].set_ylabel(metric_name, fontsize=6)
                    ax[i][j].tick_params(axis='both', which='major', labelsize=6)
                    ax[i][j].grid(True)
                    plt.tight_layout()
            # plt.show()  # Display the figure
            figures.append(fig)
        return figures
    
    # Plot weights and biases
    def plot_weights_biases(self, learning_metrics):
        print('Plotting weights and biases...')
        figures = []
        for k, model in enumerate(learning_metrics):
            if k == 1:
                break
            print(f'     Experiment {k+1}...')
            metrics = model['learning_metrics']
            network_size = len(model['model'].network) - 1
            if len(metrics) == 1:
                arch_name = 'Feedforward Neural Network'
                fig, ax = plt.subplots(2, network_size, squeeze=False, dpi = 300, constrained_layout=True)
            else:
                arch_name = 'Autoencoder + Feedforward Neural Network'
                fig, ax = plt.subplots(2, network_size, squeeze=False, dpi = 300, constrained_layout=True)
            for i, arch in enumerate(metrics):
                if len(metrics) == 2 and i == 0:
                    continue
                image_files = []
                num_epochs = list(arch['weights'].keys())
                weights_over_epochs = list(arch['weights'].values())
                biases_over_epochs = list(arch['biases'].values())
                # Calculate the maximum absolute value in the weight and bias arrays
                weights_vmax = np.zeros((len(weights_over_epochs), len(weights_over_epochs[0])))
                biases_vmax = np.zeros((len(biases_over_epochs), len(biases_over_epochs[0])))
                for e, weights_biases in enumerate(zip(weights_over_epochs, biases_over_epochs)):
                    weights, biases = weights_biases
                    for n, weight_bias in enumerate(zip(weights, biases)):
                        weight, bias = weight_bias
                        weights_vmax[e][n] = np.max(np.abs(weight))
                        biases_vmax[e][n] = np.max(np.abs(bias))
                weights_vmax = np.max(weights_vmax, axis=0)
                biases_vmax = np.max(biases_vmax, axis=0)
                # iterate over epochs and plot weights and biases for each layer
                for e, state in enumerate(zip(weights_over_epochs, biases_over_epochs)):
                    print(f'        Epoch {num_epochs[e]}...')
                    colorbars = []  # List to keep track of all colorbars
                    weights = state[0]
                    biases = state[1]
                    for n, weight_bias in enumerate(zip(weights, biases)):
                        weight, bias = weight_bias
                        im1 = ax[0][n].imshow(weight, cmap='seismic', interpolation='nearest', vmin=-weights_vmax[n], vmax=weights_vmax[n])
                        im2 = ax[1][n].imshow(bias, cmap='seismic', interpolation='nearest', vmin=-biases_vmax[n], vmax=biases_vmax[n])
                        cbar1 = plt.colorbar(im1, ax=ax[0][n])
                        # cbar1 = plt.colorbar(im1, cax=axins1)
                        cbar1.ax.tick_params(labelsize=5)
                        colorbars.append(cbar1)
                        cbar2 = plt.colorbar(im2, ax=ax[1][n])
                        # cbar2 = plt.colorbar(im2, cax=axins2)
                        cbar2.ax.tick_params(labelsize=5)
                        colorbars.append(cbar2)
                        ax[0][n].set_title('Layer ' + str(n+1) + ' Weights\n' + '[' + str(len(weight)) + ' x ' + str(len(weight[0])) + ']', fontsize=6)
                        ax[1][n].set_title('Layer ' + str(n+1) + ' Biases\n' + '[' + str(len(bias)) + ' x ' + str(len(bias[0])) + ']', fontsize=6)
                        ax[0][n].set_xlabel('Neurons in Layer ' + str(n+1), fontsize=6)
                        ax[0][n].set_ylabel('Neurons in Layer ' + str(n), fontsize=6)
                        ax[1][n].set_xlabel('Neurons in Layer ' + str(n+1), fontsize=6)
                        ax[1][n].set_ylabel('Neurons in Layer ' + str(n), fontsize=6)
                        ax[0][n].tick_params(axis='both', which='major', labelsize=6)
                        ax[1][n].tick_params(axis='both', which='major', labelsize=6)
                        ax[1][n].set_yticks([])
                    fig.suptitle('Weights & Biases for ' + arch_name + ' - Experiment ' + str(k+1) + ' - Epoch ' + str(num_epochs[e]), fontsize=8)
                    # plt.tight_layout()
                    # convert fig to image and save to temp folder for animation
                    image_path = os.path.join(self.output, 'temp', str(e) + '.png')
                    fig.savefig(image_path)
                    image_files.append(image_path)
                    # reset colormaps 
                    for cbar in colorbars:
                        cbar.remove()
                    colorbars.clear()
                # use imageio to create a GIF from the image files
                print(f'     Creating gif...')
                images = []
                for image_file in image_files:
                    images.append(imageio.imread(image_file))
                    os.remove(image_file)
                gif_path = os.path.join(self.output, 'weights_biases_' + arch_name + '_experiment_' + str(k+1) + '.gif')
                imageio.mimsave(gif_path, images, fps=15)
        #     figures.append(fig)
        # return figures
        return None
    
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
            learning_curve_name = self.model + ' learning curve with ' + self.data + 'dataset using k_x_' + str(self.splits) + ' cross validation' + '_' + str(id) + '.png'
            learning_curve_fullpath = os.path.join(self.output, learning_curve_name)
            learning_curve_fig.savefig(learning_curve_fullpath)
        if validation_curve_fig is not None:
            validation_curve_name = self.model + ' validation curve with ' + self.data + 'dataset using k_x_' + str(self.splits) + ' cross validation.png'
            validation_curve_fullpath = os.path.join(self.output, validation_curve_name)
            validation_curve_fig.savefig(validation_curve_fullpath)      
        return None