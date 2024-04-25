import pandas as pd
import numpy as np
import time
import os
import configparser
import copy
from feature_pipeline import meta_data_rl as __meta_data__

# [description] this class is responsible for transforming raw data into processed data 
# that can be used for model training or inference. It also splits the data into train,
# validation, and test sets and further into k folds for cross validation.
#
# [input] is raw data
# [output] is processed data split into train, validation, test sets. Also, 
# positive class label for binary classification is returned. 
#
class DataTransformer:
    def __init__(self, path, mode, splits, output):
        # path points to directory containing the config and data files
        self.data_path = path + '.txt'
        self.data = None # actual data 
        self.mode = mode # training or inference
        self.splits = splits # number of cross validation splits
        self.output = output # output directory
        self.start_state = 'S'
        self.goal_state = 'F'
        self.forbidden_state = '#'
    
    # Load the data file into a list of lists
    def load_data(self):
        result = []
        with open(self.data_path) as file:
            for line in file.readlines():
                if len(line) > 0:
                    result.append(list(line.strip()))
        self.data = result[1:] # skip first header line
        return None
    
    def set_meta_data(self):
        print('Setting the meta data...')
        meta_data = __meta_data__.MetaData()
        meta_data.set_meta(self.data, self.start_state, self.goal_state, self.forbidden_state)
        return meta_data

    # Run the transformer   
    def process(self):
        start_time = time.time()
        self.load_data()
        if self.mode == 'training':
            end_time = time.time()
            print(f"Data transformation time: {end_time - start_time:.2f} s")
        elif self.mode == 'inference':
            end_time = time.time()
            print(f"Data transformation time: {end_time - start_time:.2f} s")
        return self.set_meta_data()