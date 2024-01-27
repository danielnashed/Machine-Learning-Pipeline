# input is raw data 
# output is processed data split into train, validation, test sets
class DataTransformer:
    def __init__(self, config):
        # config is a path to a dataset directory containing a config file and a data file
        self.config = config + '.config'
        self.data = config + '.data'

    def load_config(self):
        # Load the config
        self.config = []
        print('Loading the config...')
        return None

    def load_data(self):
        # Transform the data
        self.data = []
        print('Loading the data...')
        return None

    def handle_missing_data(self):
        # Transform the data
        self.data = []
        print('Handling missing data...')
        return None

    def handle_outlier_data(self):
        # Transform the data
        self.data = []
        print('Handling outlier data...')
        return None

    def handle_categorical_data(self):
        # Transform the data
        self.data = []
        print('Handling categorical data...')
        return None

    def discretize_data(self):
        # Transform the data
        self.data = []
        print('Discretizing the data...')
        return None

    def transform_data(self):
        # Transform the data
        self.data = []
        print('Transforming the data...')
        return None

    def normalize_data(self):
        # Transform the data
        self.data = []
        print('Normalizing the data...')
        return None

    def standardize_data(self):
        # Transform the data
        self.data = []
        print('Standardizing the data...')
        return None

    def extract_features(self):
        # Transform the data
        self.data = []
        print('Extracting features...')
        return None

    def split_data(self):
        # Transform the data
        self.data = []
        print('Splitting the data...')
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
        return self.data