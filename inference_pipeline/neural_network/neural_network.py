import pandas as pd
import numpy as np
import copy

"""
This module contains the NeuralNetwork class which is used to create a feedforward neural network
for training and predicting target values. The neural network is built using batch gradient descent 
and the backpropagation algorithm and can be used for both classification and regression tasks. It 
also supports the use of an autoencoder neural network for learning a compressed representation of the 
input data. In addition, the class can also be used as a linear model to perform linear regression or 
logistic regression by setting the number of hidden layers to 0 in the config file.

The NeuralNetwork class contains the following attributes:
    - hyperparameters: hyperparameters for the model
    - prediction_type: classification or regression
    - function: function learned from training data
    - positive_class: positive class (if binary classification is used)
    - num_classes: number of classes (if multi-class classification is used)
    - column_names: column names for the dataset
    - classes: unique classes in the training data
    - validation_set: validation set for testing during epochs
    - clip_value: clip value for gradient clipping
    - autoencoder_layers: autoencoder hidden layers
    - is_autoencoder: is the network an autoencoder?

The NeuralNetwork class contains the following methods:
    - set_params: set the hyperparameters for the model
    - class_mappings: create mappings between class labels and integers
    - size_of_layers: calculate the number of nodes in each layer of the neural network
    - create_network: initialize the weights, biases, deltas, gradients, and activations for the neural network
    - check_over_under_flow: check if the input values are NaN or infinite
    - sigmoid: calculate the sigmoid function 
    - sigmoid_derivative: calculate the derivative of the sigmoid function
    - softmax: calculate the softmax function
    - forward_propagation: calculate the activations of the neurons in all layers of the neural network
    - back_propagation: calculate the gradient of the loss function with respect to the weights in each layer
    - l2_regularization: calculate the L2 regularization term for the weights in each layer
    - update_weights: update the weights and biases in each layer of the neural network
    - calculate_loss: calculate the loss function of the neural network
    - calculate_accuracy: calculate the accuracy of the neural network
    - calculate_mse: calculate the mean squared error of the neural network
    - train_autoencoder: train the autoencoder neural network
    - epoch_metrics: calculate the loss and metric values of the neural network at each epoch
    - run_epochs: run the training epochs of the neural network
    - fit: train the model using the training data
    - predict: make predictions using the trained model
"""

class NeuralNetwork:
    def __init__(self):
        self.hyperparameters = None # hyperparameters for the model
        self.prediction_type = None # classification or regression
        self.function = None # function learned from training data
        self.positive_class = None # positive class (if binary classification is used)
        self.num_classes = None # number of classes (if multi-class classification is used)
        self.column_names = None # column names for the dataset
        self.classes = None # unique classes in the training data
        self.validation_set = None # validation set for testing during epochs
        self.clip_value = 10 # clip value for gradient clipping
        self.autoencoder_layers = None # autoencoder hidden layers
        self.is_autoencoder = False # is the network an autoencoder?

    """
    'set_params' method is responsible for setting the hyperparameters for the model.
    Args:
        hyperparameters (dictionary): hyperparameters for the model
    Returns:
        None
    """
    def set_params(self, hyperparameters):
        self.hyperparameters = hyperparameters

    """
    'class_mappings' method is responsible for creating mappings between class labels and integers 
    to be used as unique id.
    Args:
        Y (DataFrame): target variable
    Returns:
        None
    """
    def class_mappings(self, Y):
        self.reverse_class_map = {}
        for i, cls in enumerate(Y.columns):
            self.reverse_class_map[i] = cls  # key is an integer, value is the class label
        return None

    """
    'size_of_layers' method is responsible for calculating the number of nodes in each layer of the
    neural network.
    Args:
        X (DataFrame): training data
        y (DataFrame): target variable
    Returns:
        network (list): number of nodes in each layer
    """
    def size_of_layers(self, X, y):
        # 1 - number of nodes in input layer
        input_layer_size = len(X.iloc[0]) # number of features
        # 2 - number of nodes in hidden layers
        hidden_layers_size = []
        for hyperparameter in self.hyperparameters:
            if 'hidden_layer' in hyperparameter:
                hidden_layers_size.append(self.hyperparameters[hyperparameter])
        # 3 - number of nodes in output layer
        if y is not None:
            if self.prediction_type == 'classification':
                output_layer_size = len(y.unique()) # number of classes
            elif self.prediction_type == 'regression':
                output_layer_size = 1 # single output node for regression
        else:
            # for autoencoder, output layer size is same as input layer size
            output_layer_size = input_layer_size
        return [input_layer_size] + hidden_layers_size + [output_layer_size]

    """
    'create_network' method is responsible for initializing the weights, biases, deltas, gradients, and
    activations for the neural network architecture. If the network is an autoencoder, the learned weights
    and biases inside the hidden layers from the autoencoder training are used.
    Args:
        autoencoder_function (dictionary): function learned from autoencoder training data
    Returns:
        None
    """
    def create_network(self, autoencoder_function=None):
        self.weights = [] 
        self.biases = [] 
        self.deltas = [] # error between predicted and actual values in each layer
        self.gradients = [] # gradient of the loss function with respect to the weights in each layer
        for i in range(len(self.network) - 1):
            # initialize weights for all layers to be small random values between -0.1 and 0.1
            self.weights.append(0.1 * np.random.randn(self.network[i], self.network[i+1]))
            # initialize biases for all layers to be zeros
            self.biases.append(np.zeros((1, self.network[i+1])))
            # initialize deltas for all layers
            self.deltas.append(np.zeros(self.network[i+1]))
            # initialize gradients for all layers
            self.gradients.append(np.zeros((self.network[i], self.network[i+1])))
        # if autoncoder, use the learned weights and biases 
        if autoencoder_function is not None:
            for i in range(len(autoencoder_function['weights']) - 1):
                self.weights[i] = autoencoder_function['weights'][i]
                self.biases[i] = autoencoder_function['biases'][i]
        # initialize activations for all layers
        self.activations = []
        for i in range(len(self.network)):
            self.activations.append(np.zeros(self.network[i]))
        # initialize learning rate
        self.learning_rate = self.hyperparameters['eta']
        # initialize regularization parameter
        self.l2 = self.hyperparameters['mu']
        return None

    """
    'check_over_under_flow' method is responsible for checking if the input values are NaN or infinite.
    If the input values are NaN or infinite, the method converts the values to 0 and large finite numbers
    respectively.
    Args:
        x (ndarray): input values
    Returns:
        x (ndarray): input values after checking for overflow and underflow
    """
    def check_over_under_flow(self, x):
        if np.any(np.isnan(x)) or np.any(np.isinf(x)):
            x = np.nan_to_num(x) # convert NaNs to 0 and infs to large finite numbers
        return x
    
    """
    'sigmoid' method is responsible for calculating the sigmoid function which serves as the activation 
    of the neurons in the hidden layers of the neural network. The sigmoid function is used to introduce
    non-linearity in the neural network to learn complex patterns in the data. The sigmoid function maps 
    the input values to a range between 0 and 1. The method also clips the input values to avoid overflow
    and underflow.
    Args:
        x (ndarray): input values
    Returns:
        sigmoid (ndarray): output values after applying the sigmoid function
    """
    def sigmoid(self, x):
        x = np.clip(x, -100, 100) # clip input to avoid overflow and underflow
        return 1 / (1 + np.exp(-x))
    
    """
    'sigmoid_derivative' method calculates the derivative of the sigmoid function. The derivative of the 
    sigmoid function is used to calculate the error in the hidden layers of the neural network during 
    backpropagation.
    Args:
        x (ndarray): input values
    Returns:
        sigmoid_derivative (ndarray): output values after applying the derivative of the sigmoid function
    """
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    """
    'softmax' method is calculates the softmax function which serves as the activation
    of the neurons in the output layer of the neural network. The softmax function is used to calculate
    the probabilities of the classes in the multi-class classification problem. The method also clips the
    input values to avoid overflow and underflow.
    Args:
        logits (ndarray): input values
    Returns:
        probs (ndarray): output values after applying the softmax function
    """
    def softmax(self, logits):
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True)) # exponentiate the scores
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True) # normalize the scores to get probabilities 
        return probs
    
    """
    'forward_propagation' method calculates the activations of the neurons in all layers of the neural network. 
    The method calculates the dot product of the weights and activations of the previous
    layer and adds the biases. The method then applies the sigmoid function for the hidden layers and the
    softmax function for the output layer. The method returns the output of the neural network.
    Args:
        X (ndarray): input values
    Returns:
        output (ndarray): output values at output layer of network after forward propagation
    """
    def forward_propagation(self, X):
        # activation of the input layer is the input data itself (no transformation needed)
        self.activations[0] = X
        for i in range(len(self.network) - 1):
            # net input to layer i is dot product of weights & activations of previous layer (i-1) plus biases
            dot_product = np.dot(self.activations[i], self.weights[i]) + self.biases[i]
            # calculate the activations of layer i 
            if (i == len(self.network) - 2):
                # if output layer, use softmax for classification and linear for regression
                if self.prediction_type == 'classification':
                    self.activations[i+1] = self.softmax(dot_product)
                elif self.prediction_type == 'regression':
                    self.activations[i+1] = dot_product
            else:
                self.activations[i+1] = self.sigmoid(dot_product) # use sigmoid for hidden layers
        return self.activations[-1]
    
    """
    'back_propagation' method calculates the gradient of the loss function with respect to the weights 
    in each layer. It does so by first calculating the error in the output layer between true output 
    and predicted output. It then computes the gradient as the dot product of the activations of the previous
    layer and the error delta in the output layer. 
    
    To propagate the delta error backwards through the hidden layers, the method calculates the dot product 
    of the delta error in the current layer and the weights feeding the current layer. It then scales that
    by the sigmoid function derivative using the activations of the current layer. The method also applies 
    gradient clipping to avoid exploding gradients.
    Args:
        output (ndarray): output values at output layer of network after forward propagation
        Y (ndarray): target values
    Returns:
        None
    """
    def back_propagation(self, output, Y):
        self.deltas[-1] = output - Y # error in output layer
        for i in range(len(self.network) - 2, -1, -1):
            # calculate the gradient of the loss function with respect to the weights in layer i
            self.gradients[i] = np.dot(self.activations[i].T, self.deltas[i])
            # apply gradient clipping to avoid exploding gradients
            for j in range(len(self.gradients)):
                np.clip(self.gradients[j], -self.clip_value, self.clip_value, out=self.gradients[j])
            if i == 0:
                return None
            # calculate the error in the hidden layers
            self.deltas[i-1] = np.dot(self.deltas[i], self.weights[i].T) * self.sigmoid_derivative(self.activations[i])
        return None
    
    """
    'l2_regularization' method calculates the L2 regularization term for the weights in each layer of the
    neural network. The L2 regularization term is added to the gradients of the loss function with respect
    to the weights in each layer. This helps to prevent overfitting by penalizing large weights.
    Args:
        None
    Returns:
        None
    """
    def l2_regularization(self):
        for i in range(len(self.network) - 1):
            self.gradients[i] += self.l2 * self.weights[i]
        return None

    """
    'update_weights' method updates the weights and biases in each layer of the neural network using the
    gradients of the loss function with respect to the weights in each layer. The method multiplies the
    gradients by the learning rate to update the weights. The learning rate represents the step size at 
    each iteration of weight updates. A negative term is added so that we move in the opposite direction 
    of the gradient to minimize the loss function. The method also updates the biases in each layer by 
    summing the gradients of the loss function with respect to the biases in each layer.
    Args:
        None
    Returns:
        None
    """
    def update_weights(self):
        for i in range(len(self.network) - 1):
            self.weights[i] -= self.learning_rate * self.gradients[i] # update the weights
            delta_bias = np.sum(self.gradients[i], axis=0, keepdims=True)
            self.biases[i] -= self.learning_rate * delta_bias # update the biases
        return None
    
    """
    'calculate_loss' method calculates the loss function of the neural network. The loss function is used to
    evaluate the performance of the neural network. The method calculates the cross-entropy loss for
    classification and mean squared error for regression.
    Args:
        output (ndarray): output values at output layer of network after forward propagation
        Y (ndarray): target values
    Returns:
        loss (float): loss value
    """
    def calculate_loss(self, output, Y):
        if self.prediction_type == 'classification':
            epsilon = 1e-15 
            output = np.clip(output, epsilon, 1 - epsilon) # clip output to avoid log(0)
            loss = -np.sum(Y * np.log(output), axis=0).sum() # cross entropy loss
        elif self.prediction_type == 'regression':
            loss = self.calculate_mse(output, Y)
        return loss
    
    """
    'calculate_accuracy' method calculates the accuracy of the neural network. The method calculates the 
    ratio of correctly predicted classes to total number of predictions.
    Args:
        predictions (ndarray): predicted values
        true_labels (ndarray): true values
    Returns:
        accuracy (float): accuracy value
    """
    def calculate_accuracy(self, predictions, true_labels):
        return np.mean(predictions == true_labels)
    
    """
    'calculate_mse' method calculates the mean squared error of the neural network. The method calculates the
    squared difference between the predicted values and the true values.
    Args:
        output (ndarray): output values at output layer of network after forward propagation
        Y (ndarray): target values
    Returns:
        mse (float): mean squared error value
    """
    def calculate_mse(self, output, Y):
        return np.mean((output - Y) ** 2)

    """
    'train_autoencoder' method is responsible for training the autoencoder neural network. The autoencoder
    neural network is used to learn a compressed representation of the input data. In essence, it is 
    learning to associate patterns with themselves by predicting the input at the output layer. The method 
    initializes the autoencoder neural network and trains it using batch gradient descent. The method returns 
    the trained autoencoder (its weights and biases) and the learning metrics.
    Args:
        X (DataFrame): training data
    Returns:
        autoencoder (NeuralNetwork): trained autoencoder neural network
        learning_metrics (dictionary): logs of model metric as function of epochs
    """
    def train_autoencoder(self, X):
        print('        Training autoencoder...')
        autoencoder = NeuralNetwork()
        autoencoder.hyperparameters = {'eta': 0.001, 'mu': 0.05, 'epochs': 2000}
        for layer in self.autoencoder_layers:
            num_nodes = int(self.autoencoder_layers[layer])
            # number of nodes in hidden layer is 50% of number of features in input domain
            num_nodes = int(len(X.iloc[0]) * 0.5) # this is a tunable hyperparameter
            self.autoencoder_layers[layer] = num_nodes
        autoencoder.hyperparameters.update(self.autoencoder_layers)
        autoencoder.prediction_type = 'regression' # always regression for autoencoder
        autoencoder.function = None # function learned from training data
        autoencoder.positive_class = self.positive_class # positive class (if binary classification is used)
        autoencoder.num_classes = self.num_classes # number of classes (if multi-class classification is used)
        autoencoder.column_names = self.column_names # column names for the dataset
        autoencoder.classes = self.classes # unique classes in the training data
        autoencoder.validation_set = self.validation_set # validation set for testing during epochs
        autoencoder.clip_value = self.clip_value # clip value for gradient clipping
        autoencoder.autoencoder_layers = None # flag for autoencoder
        autoencoder.is_autoencoder = True # flag for autoencoder
        autoencoder.network = autoencoder.size_of_layers(X, y=None) # calculate the size of layers
        autoencoder.create_network() # create the neural network 
        learning_metrics = autoencoder.run_epochs(X, X) # learn weights using batch gradient descent
        autoencoder.function = {'weights': autoencoder.weights, 'biases': autoencoder.biases}
        return (autoencoder, learning_metrics)
    
    """
    'epoch_metrics' method calculates the loss and metric values of the neural network at each epoch. For 
    classification, the cross-entropy loss is used along with accuracy on both training and validation
    datasets. For regression, the method uses mean square error. The method returns the weights and biases 
    of the neural network at each epoch.
    Args:
        output (ndarray): output values at output layer of network after forward propagation
        Y (ndarray): target values
    Returns:
        epoch_metrics (dictionary): logs of model metric as function of epochs
    """
    def epoch_metrics(self, output, Y):
        loss = self.calculate_loss(output, Y) # calculate loss
        Y_test_predictions = self.predict(self.validation_set[0]).values # predict on validation set
        if self.is_autoencoder:
            Y_test_labels = self.validation_set[0].values # for autoencoder, target is the input data
        else:
            Y_test_labels = self.validation_set[1].values.reshape(-1, 1)
        if self.prediction_type == 'classification':
            metric_name = 'Accuracy'
            Y_train_predictions = np.argmax(output, axis=1) # pick the class with highest probability
            Y_train_labels = np.argmax(Y, axis=1)
            training_metric = self.calculate_accuracy(Y_train_predictions, Y_train_labels)
            validation_metric = self.calculate_accuracy(Y_test_predictions, Y_test_labels)
        elif self.prediction_type == 'regression':
            metric_name = 'MSE'
            training_metric = self.calculate_mse(output, Y)
            validation_metric = self.calculate_mse(Y_test_predictions, Y_test_labels)
        return {'loss': loss, 
                'metric_name': metric_name, 
                'training': training_metric, 
                'validation': validation_metric,
                'weights': self.weights,
                'biases': self.biases}

    """
    'run_epochs' method is responsible for running the training epochs of the neural network. One epoch 
    consists of propagating the entire training input dataset fowrward to the output layer to generate
    predictions, then backpropagating the errors back to the input layer, calculating the gradient of the 
    loss function with respect to the weights in each layer along the way. The method also applies L2 
    regularization to the gradients to reduce overfitting and updates the weights using the gradients and
    learning rate. The method returns logs of metrics as function of epochs.
    Args:
        X (DataFrame): training data
        Y (DataFrame): target variable
    Returns:
        logs (dictionary): logs of model metric as function of epochs
    """
    def run_epochs(self, X, Y):
        loss_arr, metric_arr, weights_arr, biases_arr = {}, {}, {}, {}
        for epoch in range(self.hyperparameters['epochs'] + 1):
            output = self.forward_propagation(X)
            self.back_propagation(output, Y)
            self.l2_regularization() # apply L2 regularization to gradients
            self.update_weights() # update weights using gradients
            # calculate loss and metric values at each 100-th epoch
            if epoch % 100 == 0 and epoch != 0:
                # self.learning_rate = self.learning_rate * 0.9 # decrease learning rate by 10% every 100 epochs
                epoch_metrics = self.epoch_metrics(output, Y)
                loss = epoch_metrics['loss']
                metric_name = epoch_metrics['metric_name']
                training_metric = epoch_metrics['training']
                validation_metric = epoch_metrics['validation']
                loss_arr[epoch] = loss
                metric_arr[epoch] = (training_metric, validation_metric)
                weights_arr[epoch] = copy.deepcopy(self.weights)
                biases_arr[epoch] = copy.deepcopy(self.biases)
                print(f'            Epoch: {epoch} --> Loss: {loss:.5f}, Training {metric_name}: {training_metric:.5f}, Validation {metric_name}: {validation_metric:.5f}')
        return {'Loss': loss_arr, metric_name: metric_arr, 'weights': weights_arr, 'biases': biases_arr}

    """
    'fit' method is responsible for training the model using the training data. The method first checks if
    the model requires initial training of an autoencoder. If it does, the method trains the autoencoder. The 
    method then creates the feedforward neural network by dropping the output layer of the autoencoder and 
    clipping the remaining layers to the first hidden layer of the feedforward network. 
    
    This has the effect of placing the feedforward network in a more appropriate weights and biases space where the initial shallow
    layers have already learned valuable features so the weights in those shallow layers dont need alot of 
    updating during gradient descent. This helps compat the side-effects of the vanishing gradient phenomena
    becuase we focus our updates to those weights in the deep layers. Weights in the shallow layers only
    undergo fine-tuning. The method then trains the feedforward neural network using batch gradient descent. 
    The method returns logs of metrics as function of epochs. 
    Args:
        X (DataFrame): training data
        y (DataFrame): target variable
    Returns:
        learning_metrics (dictionary): logs of model metric as function of epochs
    """
    def fit(self, X, y):
        if self.autoencoder_layers is not None:
            autoencoder, autoencoder_learning_metrics = self.train_autoencoder(X)
        if self.prediction_type == 'classification':
            Y = pd.get_dummies(y) # one-hot encoding of target
            self.class_mappings(Y) # create class mappings
            Y = Y.values # pass only the values
        elif self.prediction_type == 'regression':
            Y = y.values.reshape(-1, 1) # pass only the values
        self.network = self.size_of_layers(X, y) # calculate the size of layers
        if self.autoencoder_layers is not None:
            # clip all autoencoder layers but output layer to the hidden layers of feedforward network
            self.network = autoencoder.network[:-1] + self.network[1:]
            self.create_network(autoencoder.function) # create the neural network
        else:
            self.create_network()
        print('        Training feedforward network...')
        learning_metrics = self.run_epochs(X, Y) # learn the weights using batch gradient descent 
        if self.autoencoder_layers is not None:
            learning_metrics = [autoencoder_learning_metrics, learning_metrics]
        else:
            learning_metrics = [learning_metrics]
        self.function = {'weights': self.weights, 'biases': self.biases}
        return learning_metrics

    """
    'predict' method is responsible for making predictions using the trained model. The method first
    calculates the activations of the neurons in all layers of the neural network using the forward
    propagation method. The method then returns the predicted values. For classification, the method
    returns the class with the highest probability. For regression, the method returns the predicted
    values.
    Args:
        X (DataFrame): input values
    Returns:
        y_pred (DataFrame): predicted values
    """
    def predict(self, X):
        output = self.forward_propagation(X)
        if self.prediction_type == 'classification':
            y_pred = np.argmax(output, axis=1)
            # use the reverse mapping to find the original labels
            y_pred = [self.reverse_class_map[i] for i in y_pred]
        elif self.prediction_type == 'regression':
            y_pred = output
        y_pred = pd.DataFrame(y_pred) # convert to dataframe object
        return y_pred