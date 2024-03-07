import os
import importlib
import inspect
import configparser

# [description] this class is responsible for selecting the model and loading its configuration file. It
# serves as an abstract interface to the model and sets the model as a classifier or regressor and 
# sets the positive class for classification models.
#
# [input] is directory path and positive class
# [output] is instantiated model class and its config file
#
class Model:
    def __init__(self, directory_path, meta_data):
        # get the model name from the directory path
        model_name = os.path.basename(directory_path)
        # get the model path relative to directory of this file
        model_path = f"inference_pipeline.{model_name}.{model_name}"
        # import the model using dynamic import mechanism
        imp = importlib.import_module(model_path)
        # members contains all classes and methods in the module
        module_members = inspect.getmembers(imp)
        # extract the class from the module
        module_class = next((member for member in module_members if inspect.isclass(member[1])), None)
        if module_class:
            _, self.module_class_object = module_class
        # get the model config file path 
        config_path = os.path.join(directory_path, model_name + '.config')
        self.config = self.load_config(config_path)
        self.model = None # model
        self.positive_class = meta_data['pos_class'] # positive class for binary classification
        self.num_classes = meta_data['num_classes'] # number of classes for multi-class classification
        self.column_names = meta_data['column_names'] # column names for the dataset

    # Load the config file for the model
    def load_config(self, config_path):
        print('Loading config file for model...')
        config = configparser.ConfigParser() # Create a ConfigParser object to read INI file 
        config.read(config_path)
        return config
    
    # Select the model and configure the model based on the config file
    def select(self):
        # instantiate the model class
        self.model = self.module_class_object()
        # set as classifier or regressor 
        prediction_types = dict(self.config.items('prediction_type'))
        if int(prediction_types['classification']) == 1:
            self.model.prediction_type = 'classification'
        elif int(prediction_types['regression']) == 1:
            self.model.prediction_type = 'regression'
        # set the positive class and number of classes
        self.model.positive_class = self.positive_class
        self.model.num_classes = self.num_classes
        self.model.column_names = self.column_names
        # only for decision trees, set the pruning to True or False
        if self.model.__class__.__name__ == 'DecisionTree':
            self.model.pruning = bool(int(self.config['pruning']['pruning']))
        print('Setting model to ' + self.model.__class__.__name__ + '...')
        return (self.model, self.config)