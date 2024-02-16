import pandas as pd
import numpy as np
import csv
import os
import configparser
import copy

# input is raw data 
# output is processed data split into train, validation, test sets
class DataTransformer:
    def __init__(self, path, mode, splits):
        # config is a path to a dataset directory containing a config file and a data file
        self.config_path = path + '.config'
        self.data_path = path + '.data'
        self.config = None
        self.data = None
        self.mode = mode
        self.splits = splits
        self.positive_class = None

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
        # Remove header row if it exists
        column_names = self.config.items('column_names')
        column_names = [line[1] for line in column_names]
        if self.data.iloc[0].values.tolist() == column_names:
            self.data = self.data.iloc[1:]
        # Move target column to the last column if it is not already
        target_col_index = self.config.items('target_column')[-1][1]
        if target_col_index != str(len(self.data.columns) - 1):
            self.data = self.data[[col for col in self.data.columns if col != int(target_col_index)] + [int(target_col_index)]]
            # Relabel the column names
            self.data.columns = range(len(self.data.columns))
        print('Loading the data...')
        return None
    
    def set_poitive_class(self):
        self.positive_class = self.config.items('positive_class')[0][1].strip()
        print('Setting the positive class...')
        return None

    def remove_features(self):
        cols_2_remove = self.config.items('remove_features')
        # extract the column indices to remove
        for col in cols_2_remove:
            if col[1] == '1':
                self.data = self.data.drop(columns=int(col[0])) # column names are preserved
        print('Removing features...')
        return None

    # Handle missing data
    def handle_missing_data(self):
        column_types = self.config.items('column_types')
        missing_values = self.config.items('missing_values')
        missing_values = [line[1] for line in missing_values if line[1] != '']
        for column, column_type in column_types:
            # check if column exists in dataframe 
            if int(column) not in self.data.columns:
                continue
            # if the column is a category type, fill with mode of column
            if column_type.strip().split()[0] == 'categorical':
                # if column contains missing values, replace with NaN
                if len(missing_values) != 0:
                    self.data[int(column)] = self.data[int(column)].replace(missing_values, np.nan)
                # fill NaNs with mode of column
                self.data[int(column)] = self.data[int(column)].fillna(self.data[int(column)].mode()[0])
            # if the column is a number type, fill with mean of column
            elif column_type.strip() == 'numerical':
                if len(missing_values) != 0:
                    self.data[int(column)] = self.data[int(column)].replace(missing_values, np.nan)
                self.data[int(column)] = pd.to_numeric(self.data[int(column)])
                self.data[int(column)] = self.data[int(column)].fillna(self.data[int(column)].mean())
        print('Handling missing data...')
        return None

    # To be implemented later
    def handle_outlier_data(self):
        print('Handling outlier data...')
        return None

    # convert categorical data to numeric discrete data
    def handle_categorical_data(self):
        column_types = self.config.items('column_types')
        column_encodings = self.config.items('column_encodings')
        target_column_index = int(self.config.items('target_column')[-1][1])
        # handle all categorical columns in data except target column
        for i, column in column_types[:(target_column_index - len(column_types))]:
            # check if column exists in dataframe 
            if int(i) not in self.data.columns:
                continue
            # if the column is a categorical ordinal type, encode using integer encoding
            if column == 'categorical ordinal':
                self.data[int(i)] = self.data[int(i)].replace(eval(column_encodings[int(i)][1]))
                self.data[int(i)] = self.data[int(i)].astype(float)  # cast to numeric type
            # if the column is a categorical nominal type, encode using one-hot encoding
            elif column == 'categorical nominal':
                self.data = pd.get_dummies(self.data, columns=[int(i)], dtype=float)
        # reorder columns to match the order in the config file only if hot encoding was used
        if  any('categorical nominal' in tuple for tuple in column_types):
            self.reorder_columns()
        print('Handling categorical data...')
        return None

    def reorder_columns(self):
        column_truncated_names = [str(col_name).split('_') for col_name in self.data.columns.tolist()]
        column_left_names = [int(col_name[0]) for col_name in column_truncated_names]
        column_right_names = ['_' + col_name[1] if len(col_name) == 2 else '' for col_name in column_truncated_names]
        # zip the two lists together and sort by values in the first list
        sorted_pairs = sorted(zip(column_left_names, column_right_names))
        # unzip the sorted pairs back into two lists
        column_left_names, column_right_names = zip(*sorted_pairs)
        # combine the two lists back into a list of strings
        column_names = [str(left) + right for left, right in zip(column_left_names, column_right_names)]
        # convert names of numerical columns from string to int
        column_names = [int(col) if col.isnumeric() else col for col in column_names]
        self.data = self.data[column_names] 

    # convert continuous data to discrete data
    def discretize_data(self):
        discretize_types = self.config.items('discretization')
        column_types = self.config.items('column_types')
        target_column_index = int(self.config.items('target_column')[-1][1])
        last_feature_index = target_column_index - len(column_types)
        # handle all columns in data except target column
        for discretize_section, column_section in zip(discretize_types[:last_feature_index], column_types[:last_feature_index]):
            col, discretize_type = discretize_section
            _, column_type = column_section
            # check if column exists in dataframe 
            if int(col) not in self.data.columns:
                continue
            # if column has no discretization type or is categorical, skip
            if len(discretize_type) == 0 or column_type.split()[0] == 'categorical':
                continue
            # for equal width, split into bins of equal width and use integer encoding for each bin (set labels=False)
            if list(eval(discretize_type).keys())[0] == 'equal width':
                self.data[int(col)] = pd.cut(self.data[int(col)], bins=eval(discretize_type)['equal width'], labels=False)
            # for equal frequency, split into bins of equal frequency
            elif list(eval(discretize_type).keys())[0] == 'equal frequency':
                self.data[int(col)] = pd.qcut(self.data[int(col)], q=eval(discretize_type)['equal frequency'], labels=False)
        print('Discretizing the data...')
        ## NEED MORE TESTING FOR FREQUENCY DISCRETIZATION
        return None
    
    def random_split(self, data, train_frac):
        train_data = data.sample(frac=train_frac)
        validation_data = data.drop(train_data.index)
        return [train_data, validation_data]
    
    def stratify_split(self, data, train_frac):
        # get class label distribution from config file
        class_label_distribution = self.config.items('class_distribution')
        train_data = pd.DataFrame()
        validation_data = pd.DataFrame()
        for class_label, _ in class_label_distribution:
            class_data = data[data.iloc[:, -1].astype('str') == class_label] # get all rows with class label
            train_class_data = class_data.sample(frac=train_frac) # frac of class data goes to train
            validation_class_data = class_data.drop(train_class_data.index) # remaining (1-frac) goes to validation
            train_data = pd.concat([train_data, train_class_data]) # add train data to train set
            validation_data = pd.concat([validation_data, validation_class_data]) # add validation data to validation set
        # shuffle the data to ensure randomness in order in which target classes appear
        train_data = train_data.sample(frac=1).reset_index(drop=True)
        validation_data = validation_data.sample(frac=1).reset_index(drop=True)
        return [train_data, validation_data]

    # Split the data into train and validation sets
    def split_data(self):
        target_column_index = int(self.config.items('target_column')[-1][1])
        target_class_type = self.config.items('column_types')[target_column_index][1]
        if target_class_type.split()[0] == 'categorical':
            self.data = self.stratify_split(self.data, train_frac = 0.8)
        else:
            self.data = self.random_split(self.data, train_frac = 0.8)
        print('Splitting the data...')
        return None

    # split the data into train and test data with stratification
    def split_into_k_folds(self, data):
        k_folds = []
        target_column_index = int(self.config.items('target_column')[-1][1])
        target_class_type = self.config.items('column_types')[target_column_index][1]
        if target_class_type.split()[0] == 'categorical':
            for k in range(self.splits):
                k_folds.append(self.stratify_split(data, train_frac = 0.5))
        else:
            for k in range(self.splits):
                k_folds.append(self.random_split(data, train_frac = 0.5))
        print('Splitting into k folds...')
        return k_folds
    
    def data_for_hyperparameter_tuning(self):
        train_data, validation_data = self.data
        two_halves_data = self.split_into_k_folds(train_data)
        train_validation_data = []
        for k in range(self.splits):
            train_validation_data.append([two_halves_data[k][0], copy.deepcopy(validation_data)])
            train_validation_data.append([two_halves_data[k][1], copy.deepcopy(validation_data)])
        print('Getting data for hyperparameter tuning...')
        return train_validation_data

    def data_for_model_training(self):
        train_data, _ = self.data
        two_halves_data = self.split_into_k_folds(train_data)
        train_test_data = []
        for k in range(self.splits):
            train_test_data.append([two_halves_data[k][0], two_halves_data[k][1]])
            train_test_data.append([two_halves_data[k][1], two_halves_data[k][0]])
        print('Getting data for model training...')
        return train_test_data

    # Transform the data - either normalize or standardize
    def transform_data(self, train_test_data):
        transform_types = self.config.items('transformation')
        column_types = self.config.items('column_types')
        train_test_transformed_data_outer = train_test_data
        for transform_section, column_section in zip(transform_types, column_types):
            col, transform_type = transform_section
            _, column_type = column_section
            # if column has no transformation type or is categorical type, skip
            if len(transform_type) == 0 or column_type.split()[0] == 'categorical':
                continue
            train_test_transformed_data_inner = []
            for fold in train_test_transformed_data_outer:
                train_data, test_data = fold
                # for normalization, scale the data to be between 0 and 1
                if transform_type == 'normalization':
                    train_data_min = train_data[int(col)].min()
                    train_data_max = train_data[int(col)].max()
                    train_data[int(col)] = (train_data[int(col)] - train_data_min) / (train_data_max - train_data_min)
                    test_data[int(col)] = (test_data[int(col)] - train_data_min) / (train_data_max - train_data_min)
                # for standardization, scale the data to have mean 0 and standard deviation 1
                elif transform_type == 'standardization':
                    train_data_mean = train_data[int(col)].mean()
                    train_data_std = train_data[int(col)].std()
                    train_data[int(col)] = (train_data[int(col)] - train_data_mean) / train_data_std
                    test_data[int(col)] = (test_data[int(col)] - train_data_mean) / train_data_std
                elif transform_type == 'log':
                    train_data[int(col)] = np.log(train_data[int(col)] + 1)
                    test_data[int(col)] = np.log(test_data[int(col)] + 1)
                train_test_transformed_data_inner.append([train_data, test_data])
            train_test_transformed_data_outer = train_test_transformed_data_inner
        print('Transforming the data...')
        return train_test_transformed_data_outer

    # Transform the data - either normalize or standardize
    def transform_inference_data(self, data):
        transform_types = self.config.items('transformation')
        column_types = self.config.items('column_types')
        for transform_section, column_section in zip(transform_types, column_types):
            col, transform_type = transform_section
            _, column_type = column_section
            # if column has no transformation type or is categorical type, skip
            if len(transform_type) == 0 or column_type.split()[0] == 'categorical':
                continue
            # for normalization, scale the data to be between 0 and 1
            if list(eval(transform_type).keys())[0] == 'normalization':
                data_min = data[int(col)].min()
                data_max = data[int(col)].max()
                data[int(col)] = (data[int(col)] - data_min) / (data_max - data_min)
            # for standardization, scale the data to have mean 0 and standard deviation 1
            elif list(eval(transform_type).keys())[0] == 'standardization':
                data_mean = data[int(col)].mean()
                data_std = data[int(col)].std()
                data[int(col)] = (data[int(col)] - data_mean) / data_std
        print('Transforming the data...')
        return data

    def get_features_labels(self, data):
        for i in range(len(data)):
            train_data, test_data = data[i]
            train_features = train_data.iloc[:, :-1]
            train_labels = train_data.iloc[:, -1]
            test_features = test_data.iloc[:, :-1]
            test_labels = test_data.iloc[:, -1]
            data[i] = [train_features, train_labels, test_features, test_labels]
        print('Getting features and labels...')
        return data
    
    def extract_features(self):
        # To be implemented later
        print('Extracting features...')
        return None
    
    # Export the data
    def export_data(self):
        # get directory 
        directory_path = os.path.dirname(self.data_path)
        self.data[0].to_csv(directory_path + '/processed_train_data.csv')
        self.data[1].to_csv(directory_path + '/processed_validation_data.csv')
        print('Exporting the data...')
        return None

    # Run the transformer   
    def process(self):
        self.load_config()
        self.load_data()
        self.set_poitive_class()
        self.remove_features()
        self.handle_missing_data()
        self.handle_outlier_data()
        self.handle_categorical_data()
        self.discretize_data()
        if self.mode == 'training':
            self.split_data()
            train_validation_data = self.data_for_hyperparameter_tuning()
            train_validation_data = self.transform_data(train_validation_data)
            train_validation_data = self.get_features_labels(train_validation_data)
            train_test_data = self.data_for_model_training()
            train_test_data = self.transform_data(train_test_data)
            train_test_data = self.get_features_labels(train_test_data)
            self.export_data()
            self.data = [train_validation_data, train_test_data]
            return (self.data, self.positive_class)
        elif self.mode == 'inference':
            self.data = self.transform_inference_data(self.data)
            return (self.data, self.positive_class)
        return None