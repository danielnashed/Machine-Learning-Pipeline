import os
import importlib
import inspect
import configparser

# input is inference data
# output is predictions
class Model:
    def __init__(self, directory_path, positive_class):
        model_name = os.path.basename(directory_path) # get the model name
        model_path = f"inference_pipeline.{model_name}.{model_name}" # get the model path relative to directory of this file
        imp = importlib.import_module(model_path) # import the model
        module_members = inspect.getmembers(imp)
        module_class = next((member for member in module_members if inspect.isclass(member[1])), None)
        if module_class:
            _, self.module_class_object = module_class
        config_path = os.path.join(directory_path, model_name + '.config') # setup model based on config file proided 
        self.config =  self.load_config(config_path)
        self.model = None # model
        self.positive_class = positive_class

    def load_config(self, config_path):
        # Create a ConfigParser object
        config = configparser.ConfigParser()
        # Read the config file
        config.read(config_path) 
        print('Loading config file for model...')
        return config
    
    # Select the model and configure the model based on the config file
    def select(self):
        # instantiate the model
        self.model = self.module_class_object()
        # set as classifier or regressor 
        prediction_types = dict(self.config.items('prediction_type'))
        if int(prediction_types['classification']) == 1:
            self.model.prediction_type = 'classification'
        elif int(prediction_types['regression']) == 1:
            self.model.prediction_type = 'regression'
        # set the positive class
        self.model.positive_class = self.positive_class
        print('Setting model to ' + self.model.__class__.__name__ + '...')
        return (self.model, self.config)

    # Predict
    def predict(self, data):
        predictions = []
        print('Predicting...')
        return predictions