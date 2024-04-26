import os
import ast
import itertools
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
        self.start_state = meta_data['start_state'] # start state for agent
        self.goal_state = meta_data['goal_state'] # goal state for agent
        self.forbidden_state = meta_data['forbidden_state'] # forbidden state for agent

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
        # set algorithm for engine to run reinforcement learning
        engines = dict(self.config.items('engine'))
        if int(engines['value_iteration']) == 1:
            self.model.engine = 'value_iteration'
        elif int(engines['q_learning']) == 1:
            self.model.engine = 'q_learning'
        elif int(engines['sarsa']) == 1:
            self.model.engine = 'sarsa'
        # set algorithm for crash detection
        crash = dict(self.config.items('crash_algorithm'))
        if int(crash['soft']) == 1:
            self.model.crash_algorithm = 'soft'
        elif int(crash['harsh']) == 1:
            self.model.crash_algorithm = 'harsh'
        # set start, goal and forbidden states
        self.model.start_state = self.start_state
        self.model.goal_state = self.goal_state
        self.model.forbidden_state = self.forbidden_state
        # set other model parameters
        model_parameters = dict(self.config.items('model'))
        actions = list(map(int, model_parameters['actions'].replace(' ', '').replace('[', '').replace(']', '').split(','))) # permissable actions
        # self.model.actions = list(itertools.combinations(actions, 2)) # all possible actions in form (dxx, dyy)
        self.model.actions = list(itertools.product(actions, actions)) # all possible actions in form (dxx, dyy)
        self.model.costs = ast.literal_eval(model_parameters['costs']) # costs for each state
        self.model.transition = ast.literal_eval(model_parameters['transition']) # transition probabilities
        self.model.velocity_limit = list(map(int, model_parameters['velocity_limit'].replace(' ', '').replace('[', '').replace(']', '').split(',')))
        self.model.training_iterations = int(model_parameters['training_iterations']) # number of training iterations
        self.model.reward = int(model_parameters['reward']) # reward for goal state
        self.model.alpha = float(model_parameters['alpha']) # learning rate
        self.model.gamma = float(model_parameters['gamma']) # discount factor
        print('Setting model to ' + self.model.__class__.__name__ + '...')
        return (self.model, self.config)
