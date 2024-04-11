import pandas as pd
import numpy as np

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
        self.clip_value = 1 # clip value for gradient clipping
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

    def class_mappings(self, Y):
        # self.class_map = {}
        # self.reverse_class_map = {}
        # for i, cls in enumerate(data['target'].unique()):
        #     self.class_map[cls] = i
        #     self.reverse_class_map[i] = cls
        # return None
        self.reverse_class_map = {}
        for i, cls in enumerate(Y.columns):
            self.reverse_class_map[i] = cls
        return None

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
                output_layer_size = 1
        else:
            output_layer_size = input_layer_size
        return [input_layer_size] + hidden_layers_size + [output_layer_size]

    def create_network(self, autoencoder_function=None):
        self.weights = []
        self.deltas = []
        self.gradients = []
        for i in range(len(self.network) - 1):
            # initialize weights for all layers
            self.weights.append(np.random.randn(self.network[i], self.network[i+1]))
            # initialize deltas for all layers
            self.deltas.append(np.zeros(self.network[i+1]))
            # initialize gradients for all layers
            self.gradients.append(np.zeros((self.network[i], self.network[i+1])))
        # initialize biases for all layers (except input layer)
        self.biases = []
        for i in range(1, len(self.network)):
            self.biases.append(np.zeros((1, self.network[i])))
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

    def check_over_under_flow(self, x):
        if np.any(np.isnan(x)) or np.any(np.isinf(x)):
            x = np.nan_to_num(x)
        return x
    
    def sigmoid(self, x):
        # clip input to avoid overflow and underflow
        x = np.clip(x, -100, 100)
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def softmax(self, logits):
        # Compute probabilities using softmax function
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return probs
    
    def forward_propagation(self, X):
        self.activations[0] = X
        for i in range(len(self.network) - 1):
            # calculate the dot product of weights and activations
            # self.activations[i] = self.check_over_under_flow(self.activations[i])
            # self.weights[i] = self.check_over_under_flow(self.weights[i])
            # self.biases[i] = self.check_over_under_flow(self.biases[i])
            dot_product = np.dot(self.activations[i], self.weights[i]) + self.biases[i]
            # dot_product = self.check_over_under_flow(dot_product)
            # calculate the activations of the next layer
            if (i == len(self.network) - 2):
                if self.is_autoencoder == False:
                    # if output layer, use softmax for classification and linear for regression
                    if self.prediction_type == 'classification':
                        self.activations[i+1] = self.softmax(dot_product)
                    elif self.prediction_type == 'regression':
                        self.activations[i+1] = dot_product
                else:
                    # self.activations[i+1] = self.sigmoid(dot_product)
                    self.activations[i+1] = dot_product
            else:
                self.activations[i+1] = self.sigmoid(dot_product)
        return self.activations[-1]
    
    def back_propagation(self, output, Y):
        # calculate the error in the output layer
        self.deltas[-1] = output - Y
        for i in range(len(self.network) - 2, -1, -1):
            # calculate the gradient of the weights
            # self.activations[i] = self.check_over_under_flow(self.activations[i])
            # self.deltas[i] = self.check_over_under_flow(self.deltas[i])
            # self.weights[i] = self.check_over_under_flow(self.weights[i])

            self.gradients[i] = np.dot(self.activations[i].T, self.deltas[i])
            # self.gradients[i] = self.check_over_under_flow(self.gradients[i])

            # after calculating gradients, apply gradient clipping
            for j in range(len(self.gradients)):
                # mean = np.mean(self.gradients[j])
                # print(f'            Mean of gradients: {j} --> {mean}')
                np.clip(self.gradients[j], -self.clip_value, self.clip_value, out=self.gradients[j])

            # calculate the error in the hidden layers
            # self.deltas[i] = np.dot(self.deltas[i+1], self.weights[i].T) * self.sigmoid_derivative(self.activations[i])
            if i == 0:
                return None
            self.deltas[i-1] = np.dot(self.deltas[i], self.weights[i].T) * self.sigmoid_derivative(self.activations[i])

        return None
    
    def l2_regularization(self):
        for i in range(len(self.network) - 1):
            # self.weights[i] = self.check_over_under_flow(self.weights[i])
            # self.gradients[i] = self.check_over_under_flow(self.gradients[i])
            self.gradients[i] += self.l2 * self.weights[i]
        return None

    def update_weights(self):
        for i in range(len(self.network) - 1):
            # update the weights
            # self.gradients[i] = self.check_over_under_flow(self.gradients[i])
            # self.weights[i] = self.check_over_under_flow(self.weights[i])
            self.weights[i] -= self.learning_rate * self.gradients[i]
            # update the biases
            # self.deltas[i] = self.check_over_under_flow(self.deltas[i])
            # delta_bias = np.sum(self.deltas[i], axis=0, keepdims=True)
            delta_bias = np.sum(self.gradients[i], axis=0, keepdims=True)  #try gradients instead to update bias
            # delta_bias = self.check_over_under_flow(delta_bias)
            # self.biases[i] = self.check_over_under_flow(self.biases[i])
            self.biases[i] -= self.learning_rate * delta_bias
            # self.biases[i] = self.check_over_under_flow(self.biases[i])
        return None
    
    def calculate_loss(self, output, Y):
        epsilon = 1e-15
        output = np.clip(output, epsilon, 1 - epsilon)
        loss = -np.sum(Y * np.log(output), axis=0).sum() # cross entropy loss
        return loss
    
    def calculate_accuracy(self, predictions, true_labels):
        # predictions = np.argmax(output, axis=1)
        # true_labels = np.argmax(Y, axis=1)
        accuracy = np.mean(predictions == true_labels)
        return accuracy
    
    def calculate_mse(self, output, Y):
        mse = np.mean((output - Y) ** 2)
        return mse

    def train_autoencoder(self, X):
        print('        Training autoencoder...')
        autoencoder = NeuralNetwork()
        autoencoder.hyperparameters = {'eta': 0.01, 'mu': 0.05, 'epochs': 1000}
        for layer in self.autoencoder_layers:
            num_nodes = int(self.autoencoder_layers[layer])
            if num_nodes >= len(X.iloc[0]):
                num_nodes = int(len(X.iloc[0]) * 0.25) # reduce the number of nodes to be less than the number of features
            self.autoencoder_layers[layer] = num_nodes
        autoencoder.hyperparameters.update(self.autoencoder_layers)
        autoencoder.prediction_type = 'regression' # classification or regression
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
        # learn the weights using batch gradient descent 
        learning_metrics = autoencoder.run_epochs(X, X)
        autoencoder.function = {'weights': autoencoder.weights, 'biases': autoencoder.biases}
        return (autoencoder, learning_metrics)
    
    def epoch_metrics(self, output, Y):
        loss = self.calculate_loss(output, Y) # calculate loss
        Y_test_predictions = self.predict(self.validation_set[0]).values
        if self.is_autoencoder:
            Y_test_labels = self.validation_set[0].values
        else:
            Y_test_labels = self.validation_set[1].values.reshape(-1, 1)
        if self.prediction_type == 'classification':
            metric_name = 'Accuracy'
            Y_train_predictions = np.argmax(output, axis=1) 
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

    def run_epochs(self, X, Y):
        loss_arr, metric_arr, weights_arr, biases_arr = {}, {}, {}, {}
        for epoch in range(self.hyperparameters['epochs'] + 1):
            output = self.forward_propagation(X)
            self.back_propagation(output, Y)
            self.l2_regularization()
            self.update_weights()
            ### debug
            flag = any(np.isnan(weight_matrix).any() for weight_matrix in self.weights)
            if flag or epoch == 600:
                debug = ''
            if epoch % 100 == 0 and epoch != 0:
                epoch_metrics = self.epoch_metrics(output, Y)
                loss = epoch_metrics['loss']
                metric_name = epoch_metrics['metric_name']
                training_metric = epoch_metrics['training']
                validation_metric = epoch_metrics['validation']
                loss_arr[epoch] = loss
                metric_arr[epoch] = (training_metric, validation_metric)
                weights_arr[epoch] = self.weights
                biases_arr[epoch] = self.biases
                print(f'            Epoch: {epoch} --> Loss: {loss:.5f}, Training {metric_name}: {training_metric:.5f}, Validation {metric_name}: {validation_metric:.5f}')
        return {'Loss': loss_arr, metric_name: metric_arr}

    """
    'fit' method is responsible for training the model.
    Args:
        X (DataFrame): training data
        y (DataFrame): target variable
    Returns:
        learning_metrics (dictionary): logs of model metric as function of % of training data
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
            self.network = autoencoder.network[:-1] + self.network[1:]
            self.create_network(autoencoder.function) # create the neural network
        else:
            self.create_network() # create the neural network
        print('        Training feedforward network...')
        # learn the weights using batch gradient descent 
        learning_metrics = self.run_epochs(X, Y)
        if self.autoencoder_layers is not None:
            learning_metrics = [autoencoder_learning_metrics, learning_metrics]
        else:
            learning_metrics = [learning_metrics]
        self.function = {'weights': self.weights, 'biases': self.biases}
        return learning_metrics
    

    def linear_regression(self, data):
        num_features = len(data.iloc[0]) - 1 # number of features
        W = np.zeros(num_features) # initialize weights to zero
        learning_rate = self.hyperparameters['learning_rate'] # learning rate
        count = 0 # count number of iterations
        while True:
            count += 1
            print(f'        Iteration: {count}')
            X = data 
            Y = X['target']
            X = X.drop('target', axis=1)
            delta_W = np.dot(X.T, np.dot(X, W) - Y) # calculate gradient
            W -= learning_rate * delta_W / len(data) # update weights
            if count > 1000 or abs(np.sum(delta_W)) < 1e-6:
                break
        return W
    
    def logistic_regression(self, data):
        num_features = len(data.iloc[0]) - 1 # number of features
        W = np.zeros((num_features, self.num_classes)) # initialize weights to zero
        learning_rate = self.hyperparameters['learning_rate'] # learning rate
        count = 0 # count number of iterations
        mu = 0.01
        while True:
            count += 1
            print(f'        Iteration: {count}')
            X = data.drop('target', axis=1)
            Y = pd.get_dummies(data['target']).values # one-hot encoding of target
            logits = np.dot(X, W) # calculate scores
            # Compute probabilities using softmax function
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            delta_W = np.dot(X.T, (probs - Y)) + 2 * mu * W # calculate gradient of cross-entropy loss
            W -= learning_rate * delta_W / len(data) # update weights
            if count > 1000 or  np.linalg.norm(delta_W) < 1e-10:
                break
        return W

    """
    'predict' method is responsible for predicting the target values. The method traverses the decision
    tree to predict the target values for the query points. The method returns the predicted target
    values as a DataFrame object.
    Args:
        X (DataFrame): query points
        function (Node object): function learned from training data (decision tree in this case)
    Returns:
        y_pred (DataFrame): predicted target values
    """
    # def predict(self, X, function=None):
    #     W = self.function
    #     if self.prediction_type == 'classification':
    #         logits = np.dot(X, W)  # calculate scores
    #         exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))  # exponentiate the scores
    #         # exp_logits = np.exp(logits)  # exponentiate the scores
    #         probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)  # calculate probabilities
    #         y_pred = np.argmax(probs, axis=1)  # choose the class with the highest probability
    #     elif self.prediction_type == 'regression':
    #         y_pred = np.dot(X, W)  # calculate prediction
    #     y_pred = pd.DataFrame(y_pred) # convert to dataframe object
    #     return y_pred


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