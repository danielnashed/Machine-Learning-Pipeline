
# input is raw data 
# output is processed data split into train, validation, test sets
class Data_Transformer:
    def __init__(self, config):
        self.config = config
        self.data = None

    def load_data(self, data):
        # Transform the data
        self.data = data
        pass

    def handle_missing_data(self, data):
        # Transform the data
        self.data = data
        pass

    def handle_outlier_data(self, data):
        # Transform the data
        self.data = data
        pass

    def handle_categorical_data(self, data):
        # Transform the data
        self.data = data
        pass

    def discretize_data(self, data):
        # Transform the data
        self.data = data
        pass

    def transform_data(self, data):
        # Transform the data
        self.data = data
        pass

    def normalize_data(self, data):
        # Transform the data
        self.data = data
        pass

    def standardize_data(self, data):
        # Transform the data
        self.data = data
        pass

    def extract_features(self, data):
        # Transform the data
        self.data = data
        pass

    def split_data(self, data):
        # Transform the data
        self.data = data
        pass

    def process(self, data):
        # Run the transformer
        self.data = data
        self.load_data(self.data)
        self.handle_missing_data(self.data)
        self.handle_outlier_data(self.data)
        self.handle_categorical_data(self.data)
        self.discretize_data(self.data)
        self.normalize_data(self.data)
        self.standardize_data(self.data)
        self.extract_features(self.data)
        self.split_data(self.data)
        return self.data

# input is processed data and model
# output is model
class Learner:
    def __init__(self, config, model=None):
        self.config = config
        self.model = model

    def create_model(self, data):
        # Train the model
        pass

    def train_model(self, data):
        # Train the model
        pass

    def validate_model(self, data):
        # Validate the model
        pass

    def export_model(self, data):
        # Validate the model
        pass

    def export_logs(self, data):
        # Export the logs
        pass

    def learn(self, data):
        # Run the learner
        self.create_model(data)
        self.train_model(data)
        self.validate_model(data)
        self.export_model(data)
        return self.model


# input is inference data
# output is predictions
class Model:
    def __init__(self, config):
        self.config = config

    def predict(self, data):
        # Predict
        pass


# input is predictions and labels
# output is evaluation metrics
class Evaluator:
    def __init__(self, config):
        self.config = config

    def evaluate_model(self, labels, predictions):
        # Evaluate the model
        pass

# input is evaluation metrics
# output is dashboard visuals
class Dashboard:
    def __init__(self, config):
        self.config = config

    def visualize(self, evaluation):
        # Visualize the data
        pass

    def export_dashboard(self, data):
        # Export the dashboard
        pass

class Pipeline:
    def __init__(self, config):
        self.config = config
        self.pipeline = self._build_pipeline()

    def _build_pipeline(self):
        # Build the pipeline

        # 1. Load data
        # 2. Preprocess data
        # 3. Train model
        # 4. Evaluate model
        # 5. Save model
        # 6. Save results
        # 7. Save logs

        pass

    def run(self):
        # Run the pipeline
        pass





# Path: pipeline/pipeline.py
    
def load_config(model_name='fpn_resnet', configs=json file):
    # Load configs from json file
    pass

def main():
    config = load_config()
    pipeline = Pipeline(config)
    pipeline.run()

if __name__ == '__main__':
    main()


# def export_results(self, data):
# # Export the results
# pass


# class Logger:
#     def __init__(self, config):
#         self.config = config

#     def log(self, data):
#         # Log the data
#         pass

## the whole ml pipeline

# raw data -> transformer -> processed data (ingest raw data, transform it, and output processed data)
# processed data -> learner -> model (ingest processed data, train model, and output model)
# inference or test data -> model -> predictions 
# predictions -> evaluator -> results
# results -> logger -> logs
# logs -> dashboard -> dashboard visuals


# 1. Load data
# 2. Preprocess data
# 3. Feature engineering
# 4. model selection based on config file (model selector)
# 3. Train + validate model
# 4. Evaluate model
# 5. Save model
# 6. Save results
# 7. Save logs


# can also have a data piepline and a model pipeline
# data pipeline: load data, preprocess data, save data
# model pipeline: load data, train model, save model, save results, save logs


##
# Simple, validation is a step taken during the training process to fine-tune and validate a model on different subsets of the training data, while evaluation is the final assessment of the model's performance on completely new, unseen data to estimate its effectiveness in real-world scenarios




# evaluator should only take predictions and labels as input and output metrics