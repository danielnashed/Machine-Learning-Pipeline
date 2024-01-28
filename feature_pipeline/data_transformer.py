import pandas as pd
import numpy as np
import csv
import os
import configparser

# {
#     "name": "John Doe",
#     "age": 30,
#     "city": "New York"
# }


# input is raw data 
# output is processed data split into train, validation, test sets
class DataTransformer:
    def __init__(self, config):
        # config is a path to a dataset directory containing a config file and a data file
        self.config_path = config + '.config'
        self.data_path = config + '.data'
        self.config = None
        self.data = None

    def load_config(self):
        # Create a ConfigParser object
        config = configparser.ConfigParser()
        # Read the config file
        config.read(self.config_path) 
        self.config = config
        print('Loading the config...')
        return None

    def load_data(self):
        # Open the file in read mode
        with open(self.data_path, "r") as file:
            # Create a CSV reader object
            csv_reader = csv.reader(file)
            data = []
            # Read the contents of the file line by line 
            for line in csv_reader:
                data.append(line) # add the line to data list
        self.data = pd.DataFrame(data) # convert to dataframe object
        # Replace empty strings with NaN values, must reassign back to self.data
        self.data = self.data.replace('', np.nan)
        # Remove last row if its all empty
        if self.data.iloc[-1].isnull().any():
            self.data = self.data.iloc[:-1]
        print('Loading the data...')
        return None

    # Handle missing data
    def handle_missing_data(self):
        # convert all numeric columns to numric type 
        self.data.convert_dtypes(convert_integer=True)
        # get data type of each column
        column_types = self.data.dtypes
        for column in column_types.index:
            # if the column is a category type, fill with mode of column
            if column_types[column] == 'object':
                self.data[column] = self.data[column].fillna(self.data[column].mode()[0])
            # if the column is a number type, fill with mean of column
            elif column_types[column] == 'number':
                self.data[column] = self.data[column].fillna(self.data[column].mean())
        print('Handling missing data...')
        return None

    def handle_outlier_data(self):
        # To be implemented later
        self.data = self.data
        print('Handling outlier data...')
        return None

    def handle_categorical_data(self):
        column_types = self.config.items('column_types')
        column_encodings = self.config.items('column_encodings')
        for i, column in column_types:
            # if the column is a categorical ordinal type, encode using integer encoding
            if column == 'categorical ordinal':
                self.data[int(i)] = self.data[int(i)].replace(eval(column_encodings[int(i)][1]))
                self.data[int(i)] = self.data[int(i)].astype(float)  # cast to numeric type
                # self.data[int(i)] = self.data[int(i)].astype('category') # cast to categorical type
                # self.data[int(i)] = self.data[int(i)].cat.codes # encode using integer encoding
            # if the column is a categorical nominal type, encode using one-hot encoding
            elif column == 'categorical nominal':
                self.data = pd.get_dummies(self.data, columns=[int(i)], dtype=float)
        # reorder columns to match the order in the config file
        columns_order = sorted([str(col_name) for col_name in self.data.columns.tolist()])
        columns_order = [int(col_name) if col_name.isdigit() else col_name for col_name in columns_order]  
        self.data = self.data[columns_order] 
        print('Handling categorical data...')
        return None

    def discretize_data(self):
        discretize_types = self.config.items('discretization')
        for col, discretize_type in discretize_types:
            # if column has no discretization type, skip
            if len(discretize_type) != 0:
                # for equal width, split into bins of equal width and use integer encoding for each bin (set labels=False)
                if list(eval(discretize_type).keys())[0] == 'equal width':
                    self.data[int(col)] = pd.cut(self.data[int(col)], bins=eval(discretize_type)['equal width'], labels=False)
                # for equal frequency, split into bins of equal frequency
                elif list(eval(discretize_type).keys())[0] == 'equal frequency':
                    self.data[int(col)] = pd.qcut(self.data[int(col)], q=eval(discretize_type)['equal frequency'], labels=False)
        print('Discretizing the data...')
        return None

    def transform_data(self):
        # Transform the data
        self.data = self.data
        print('Transforming the data...')
        return None

    def normalize_data(self):
        # Transform the data
        self.data = self.data
        print('Normalizing the data...')
        return None

    def standardize_data(self):
        # Transform the data
        self.data = self.data
        print('Standardizing the data...')
        return None

    def extract_features(self):
        # Transform the data
        self.data = self.data
        print('Extracting features...')
        return None

    def split_data(self):
        # Transform the data
        self.data = self.data
        print('Splitting the data...')
        return None

    # Export the data
    def export_data(self):
        # get directory 
        directory_path = os.path.dirname(self.data_path)
        self.data.to_csv(directory_path + '/processed_data.csv')
        print('Exporting the data...')
        return None

    def process(self):
        # Run the transformer
        self.load_config()
        self.load_data()
        self.handle_missing_data()
        self.handle_outlier_data()
        self.handle_categorical_data()
        self.discretize_data()
        self.normalize_data()
        self.standardize_data()
        self.extract_features()
        self.split_data()
        self.export_data()
        return self.data