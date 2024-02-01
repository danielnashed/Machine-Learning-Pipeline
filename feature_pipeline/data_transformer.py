import pandas as pd
import numpy as np
import csv
import os
import configparser

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

    # To be implemented later
    def handle_outlier_data(self):
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
        column_types = self.config.items('column_types')
        for discretize_section, column_section in zip(discretize_types, column_types):
            col, discretize_type = discretize_section
            _, column_type = column_section
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
    
    # Split the data into train and validation sets
    def split_data(self):
        if self.mode != 'training':
            return
        train_data = self.data.sample(frac=0.8, random_state=42)
        validation_data = self.data.drop(train_data.index)
        self.data = [train_data, validation_data]
        print('Splitting the data...')
        # must return x, y for train, validation, test sets
        return None

    # split the data into train and test data with stratification
    def split_into_k_folds(self, data):
        # get class label distribution from config file
        class_label_distribution = self.config.items('class_distribution')
        column_encodings = self.config.items('column_encodings')
        k_folds = []
        for k in range(self.splits):
            train_data = pd.DataFrame()
            test_data = pd.DataFrame()
            for class_label, distribution in class_label_distribution:
                distribution = float(distribution)/100 # convert % to decimal probability 
                encoded_class_label = eval(column_encodings[-1][1])[class_label] # get integer encoding of class label
                class_data = data[data.iloc[:, -1] == encoded_class_label] # get all rows with class label
                train_class_data = class_data.sample(frac=0.5, random_state=42) # 50% of class data goes to train
                test_class_data = class_data.drop(train_class_data.index) # remaining 50% goes to test
                # train_data = train_data.concat(train_class_data) # add train data to train set
                # test_data = test_data.concat(test_class_data) # add test data to test set
                train_data = pd.concat([train_data, train_class_data]) # add train data to train set
                test_data = pd.concat([test_data, test_class_data]) # add test data to test set
            k_folds.append([train_data, test_data])
        print('Splitting into k folds...')
        return k_folds
    
    def data_for_hyperparameter_tuning(self):
        train_data, validation_data = self.data
        two_halves_data = self.split_into_k_folds(train_data)
        train_validation_data = []
        for k in range(self.splits):
            train_validation_data.append([two_halves_data[k][0], validation_data])
            train_validation_data.append([two_halves_data[k][1], validation_data])
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
            # train_data_all, test_data_all = train_test_data
            train_data_all, test_data_all = train_test_transformed_data_outer
            train_test_transformed_data_inner = []
            for train_data, test_data in zip(train_data_all, test_data_all):
                # for normalization, scale the data to be between 0 and 1
                if list(eval(transform_type).keys())[0] == 'normalization':
                    train_data_min = train_data[int(col)].min()
                    train_data_max = train_data[int(col)].max()
                    train_data[int(col)] = (train_data[int(col)] - train_data_min) / (train_data_max - train_data_min)
                    test_data[int(col)] = (test_data[int(col)] - train_data_min) / (train_data_max - train_data_min)
                # for standardization, scale the data to have mean 0 and standard deviation 1
                elif list(eval(transform_type).keys())[0] == 'standardization':
                    train_data_mean = train_data[int(col)].mean()
                    train_data_std = train_data[int(col)].std()
                    train_data[int(col)] = (train_data[int(col)] - train_data_mean) / train_data_std
                    test_data[int(col)] = (test_data[int(col)] - train_data_mean) / train_data_std
                train_test_transformed_data_inner.append([train_data, test_data])
            train_test_transformed_data_outer = train_test_transformed_data_inner
        print('Transforming the data...')
        return train_test_transformed_data_outer

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
        self.handle_missing_data()
        self.handle_outlier_data()
        self.handle_categorical_data()
        self.discretize_data()
        self.split_data()
        train_validation_data = self.data_for_hyperparameter_tuning()
        train_validation_data = self.transform_data(train_validation_data)
        train_validation_data = self.get_features_labels(train_validation_data)
        train_test_data = self.data_for_model_training()
        train_test_data = self.transform_data(train_test_data)
        train_test_data = self.get_features_labels(train_test_data)
        #self.extract_features()
        self.export_data()
        self.data = [train_validation_data, train_test_data]
        return self.data