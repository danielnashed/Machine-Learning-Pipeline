import pandas as pd
import numpy as np
import time
import csv
import os
import configparser
import copy

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
        self.config_path = path + '.config'
        self.data_path = path + '.data'
        self.config = None # actual config object
        self.data = None # actual data 
        self.mode = mode # training or inference
        self.splits = splits # number of cross validation splits
        self.output = output # output directory
        self.positive_class = None # positive class label (only for binary classification)

    # Load the config file
    def load_config(self):
        print('Loading the config...')
        config = configparser.ConfigParser() # create a ConfigParser object to detect INI file structure
        config.read(self.config_path) # read the config file
        self.config = config
        return None

    # Load the csv data file into a pandas dataframe
    def load_data(self):
        print('Loading the data...')
        # Open the file in read mode
        with open(self.data_path, "r") as file:
            csv_reader = csv.reader(file) # create a CSV reader object
            data = []
            # Read the contents of the file line by line 
            for line in csv_reader:
                data.append(line)
        self.data = pd.DataFrame(data) # convert to dataframe object
        # Replace empty strings with NaN values, must reassign back to self.data
        self.data = self.data.replace('', np.nan)
        # Remove last row if it is all empty
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
        return None
    
    # Set the positive class label for binary classification
    def set_poitive_class(self):
        print('Setting the positive class...')
        self.positive_class = self.config.items('positive_class')[0][1].strip()
        return None

    # Remove unnecessary features
    def remove_features(self):
        print('Removing features...')
        cols_2_remove = self.config.items('remove_features')
        # extract the column indices to remove
        for col in cols_2_remove:
            if col[1] == '1':
                self.data = self.data.drop(columns=int(col[0])) # column names are preserved
        return None

    # Handle missing data
    def handle_missing_data(self):
        print('Handling missing data...')
        column_types = self.config.items('column_types')
        missing_values = self.config.items('missing_values')
        # extract all symbols for missing values in dataset
        missing_values = [line[1] for line in missing_values if line[1] != '']
        for column, column_type in column_types:
            # check if column exists in dataframe 
            if int(column) not in self.data.columns:
                continue
            # if the column is a category type, fill with mode of column
            if column_type.strip().split()[0] == 'categorical':
                # if column contains missing value symbols, replace with NaN
                if len(missing_values) != 0:
                    self.data[int(column)] = self.data[int(column)].replace(missing_values, np.nan)
                # fill NaNs with mode of column
                self.data[int(column)] = self.data[int(column)].fillna(self.data[int(column)].mode()[0])
            # if the column is a number type, fill with mean of column
            elif column_type.strip() == 'numerical':
                # if column contains missing value symbols, replace with NaN
                if len(missing_values) != 0:
                    self.data[int(column)] = self.data[int(column)].replace(missing_values, np.nan)
                # convert to numeric type and fill NaNs with mean of column
                self.data[int(column)] = pd.to_numeric(self.data[int(column)])
                self.data[int(column)] = self.data[int(column)].fillna(self.data[int(column)].mean())
        return None

    # Convert categorical data to numeric discrete data
    def handle_categorical_data(self):
        print('Handling categorical data...')
        column_types = self.config.items('column_types')
        column_encodings = self.config.items('column_encodings')
        # extract index of the target column
        target_column_index = int(self.config.items('target_column')[-1][1])
        # handle all categorical columns in data except target column
        for i, column in column_types:
            # check if column exists in dataframe or if it is target column
            if (int(i) not in self.data.columns) or (int(i) == target_column_index):
                continue
            # if the column is a categorical ordinal type, encode using integer encoding
            if column == 'categorical ordinal':
                self.data[int(i)] = self.data[int(i)].replace(eval(column_encodings[int(i)][1]))
                self.data[int(i)] = self.data[int(i)].astype(float)  # cast to numeric float type
            # if the column is a categorical nominal type, encode using one-hot encoding
            elif column == 'categorical nominal':
                self.data = pd.get_dummies(self.data, columns=[int(i)], dtype=float)
        # reorder columns to match the order in the config file only if hot encoding was used
        if  any('categorical nominal' in tuple for tuple in column_types):
            self.reorder_columns()
        return None

    # Reorder columns to match the order in the config file
    def reorder_columns(self):
        # get index for target column so that it is excluded from reordering
        target_col_index = self.config.items('target_column')[-1][1]
        # split column names into two parts: left part is the index and right part is the name
        column_truncated_names = [str(col_name).split('_') for col_name in self.data.columns.tolist() if str(col_name) != target_col_index]
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
        # add back the target column to the end of the list
        column_names.append(int(target_col_index))
        self.data = self.data[column_names] 
        return None

    # Convert continuous data to discrete data
    def discretize_data(self):
        print('Discretizing the data...')
        discretize_types = self.config.items('discretization')
        column_types = self.config.items('column_types')
        # handle all columns in data except target column
        for discretize_section, column_section in zip(discretize_types, column_types):
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
        return None
    
    # Split the data randomly into train and validation sets without stratification
    def random_split(self, data, train_frac):
        train_data = data.sample(frac=train_frac)
        validation_data = data.drop(train_data.index)
        return [train_data, validation_data]
    
    # Split the data into train and validation sets with stratification
    def stratify_split(self, data, train_frac):
        # get class label distribution from config file
        class_label_distribution = self.config.items('class_distribution')
        train_data = pd.DataFrame()
        validation_data = pd.DataFrame()
        for class_label, _ in class_label_distribution:
            class_data = data[data.iloc[:, -1].astype('str') == class_label] # get all rows with class label
            train_class_data = class_data.sample(frac=train_frac) # frac of class data goes to train
            validation_class_data = class_data.drop(train_class_data.index) # remaining goes to validation
            train_data = pd.concat([train_data, train_class_data]) # concatenate train data
            validation_data = pd.concat([validation_data, validation_class_data]) # concatenate validation data
        # shuffle the data to ensure randomness in order in which target classes appear
        train_data = train_data.sample(frac=1).reset_index(drop=True)
        validation_data = validation_data.sample(frac=1).reset_index(drop=True)
        return [train_data, validation_data]

    # Split the data into train and validation sets using 80/20% rule
    def split_data(self):
        print('Splitting the data...')
        target_column_index = int(self.config.items('target_column')[-1][1])
        target_class_type = self.config.items('column_types')[target_column_index][1]
        # if target column is categorical, stratify the split
        if target_class_type.split()[0] == 'categorical':
            self.data = self.stratify_split(self.data, train_frac = 0.8)
        # if target column is numerical, split randomly
        else:
            self.data = self.random_split(self.data, train_frac = 0.8)
        return None

    # Split the data into k folds for cross validation using 50/50% rule
    def split_into_k_folds(self, data):
        print('Splitting into k folds...')
        k_folds = []
        target_column_index = int(self.config.items('target_column')[-1][1])
        target_class_type = self.config.items('column_types')[target_column_index][1]
        # if target column is categorical, stratify the split
        if target_class_type.split()[0] == 'categorical':
            for k in range(self.splits):
                k_folds.append(self.stratify_split(data, train_frac = 0.5))
        # if target column is numerical, split randomly
        else:
            for k in range(self.splits):
                k_folds.append(self.random_split(data, train_frac = 0.5))
        return k_folds
    
    # Get data for hyperparameter tuning
    def data_for_hyperparameter_tuning(self):
        print('Getting data for hyperparameter tuning...')
        train_data, validation_data = self.data
        # split train data into k x 2 folds
        two_halves_data = self.split_into_k_folds(train_data)
        train_validation_data = []
        # for each fold, use one half as train data and the other half as validation data
        for k in range(self.splits):
            train_validation_data.append([two_halves_data[k][0], copy.deepcopy(validation_data)])
            train_validation_data.append([two_halves_data[k][1], copy.deepcopy(validation_data)])
        return train_validation_data

    # Get data for model training
    def data_for_model_training(self):
        print('Getting data for model training...')
        train_data, _ = self.data
        # split train data into k x 2 folds
        two_halves_data = self.split_into_k_folds(train_data)
        train_test_data = []
        # for each fold, use one half as train data and the other half as test data
        for k in range(self.splits):
            train_test_data.append([two_halves_data[k][0], two_halves_data[k][1]])
            train_test_data.append([two_halves_data[k][1], two_halves_data[k][0]])
        return train_test_data

    # Transform the data - either normalize or standardize
    def transform_data(self, train_test_data):
        print('Transforming the data...')
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
            # for each fold, transform the data
            for fold in train_test_transformed_data_outer:
                train_data, test_data = fold
                # for normalization, scale the data to be between 0 and 1 using min-max scaling
                if transform_type == 'normalization':
                    # min and max values are computed from the train data only
                    train_data_min = train_data[int(col)].min()
                    train_data_max = train_data[int(col)].max()
                    train_data[int(col)] = (train_data[int(col)] - train_data_min) / (train_data_max - train_data_min)
                    test_data[int(col)] = (test_data[int(col)] - train_data_min) / (train_data_max - train_data_min)
                # for standardization, scale the data to have mean 0 and standard deviation 1
                elif transform_type == 'standardization':
                    # mean and standard deviation values are computed from the train data only
                    train_data_mean = train_data[int(col)].mean()
                    train_data_std = train_data[int(col)].std()
                    train_data[int(col)] = (train_data[int(col)] - train_data_mean) / train_data_std
                    test_data[int(col)] = (test_data[int(col)] - train_data_mean) / train_data_std
                # for target variables that are skewed to 0, apply log transformation
                elif transform_type == 'log':
                    train_data[int(col)] = np.log(train_data[int(col)] + 1)
                    test_data[int(col)] = np.log(test_data[int(col)] + 1)
                train_test_transformed_data_inner.append([train_data, test_data])
            train_test_transformed_data_outer = train_test_transformed_data_inner
        return train_test_transformed_data_outer

    # Transform inference data - either normalize or standardize
    def transform_inference_data(self, data):
        print('Transforming the data...')
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
        return data

    # Split the data into features and labels
    def get_features_labels(self, data):
        print('Getting features and labels...')
        for i in range(len(data)):
            train_data, test_data = data[i]
            train_features = train_data.iloc[:, :-1]
            train_labels = train_data.iloc[:, -1]
            test_features = test_data.iloc[:, :-1]
            test_labels = test_data.iloc[:, -1]
            data[i] = [train_features, train_labels, test_features, test_labels]
        return data
    
    # Export the data to csv files before transformation and k-fold splitting
    def export_data(self):
        print('Exporting the data...')
        self.data[0].to_csv(os.path.join(self.output, 'processed_train_data.csv'))
        self.data[1].to_csv(os.path.join(self.output, 'processed_validation_data.csv'))
        return None

    # Run the transformer   
    def process(self):
        start_time = time.time()
        self.load_config()
        self.load_data()
        self.set_poitive_class()
        self.remove_features()
        self.handle_missing_data()
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
            end_time = time.time()
            print(f"Data transformation time: {end_time - start_time:.2f} s")
            return (self.data, self.positive_class)
        elif self.mode == 'inference':
            self.data = self.transform_inference_data(self.data)
            end_time = time.time()
            print(f"Data transformation time: {end_time - start_time:.2f} s")
            return (self.data, self.positive_class)
        return None