## Package imports
import os
import sys
from feature_pipeline import data_transformer
from training_pipeline import learner, evaluator, dashboard
from inference_pipeline import model as model_selector


class Pipeline:
    def __init__(self, config):
        self.model, self.data, self.mode, self.splits = self.load_config(config)

    def load_config(self, config):
        model_name = config['model']
        dataset = config['dataset']
        mode = config['mode']
        splits = config['cross_validation_splits']

        ## Add current working directory to path
        root_dir = os.getcwd()
        sys.path.append(root_dir)

        ## Import the model config file
        model_fullpath = os.path.join(root_dir, 'inference_pipeline', model_name)

        ## Import the dataset config file
        if dataset == 'car':
            directory = os.path.join(root_dir, 'datasets', 'Classification Data Sets', 'Car Evaluation')
        elif dataset == 'breast-cancer-wisconsin':
            directory = os.path.join(root_dir, 'datasets', 'Classification Data Sets', 'Breast Cancer')
        elif dataset == 'house-votes-84':
            directory = os.path.join(root_dir, 'datasets', 'Classification Data Sets', 'Congressional Vote')
        elif dataset == 'abalone':
            directory = os.path.join(root_dir, 'datasets', 'Regression Data Sets', 'Abalone')
        elif dataset == 'machine':
            directory = os.path.join(root_dir, 'datasets', 'Regression Data Sets', 'Computer Hardware')
        elif dataset == 'forestfires':
            directory = os.path.join(root_dir, 'datasets', 'Regression Data Sets', 'Forest Fires')
        elif dataset == 'racetracks':
            directory = os.path.join(root_dir, 'datasets', 'Reinforcement Learning Data Sets', 'Racetracks')
        data_fullpath = os.path.join(directory, dataset) 

        return (model_fullpath, data_fullpath, mode, splits)

    # Build the pipeline
    def _build_pipeline(self):
        # 1. Load data and preprocess it
        print('*** START OF DATA TRANSFORMATION ***\n')
        data, pos_class = data_transformer.DataTransformer(self.data, self.mode, self.splits).process()
        print('\n*** END OF DATA TRANSFORMATION ***\n')
        # 2. Create the model (and train it if in training mode)
        print('*** START OF MODEL SELECTION ***\n')
        model = model_selector.Model(self.model, pos_class).select()
        print('\n*** END OF MODEL SELECTION ***\n')
        if self.mode == 'training':
            print('*** START OF MODEL TRAINING ***\n')
            model, logs = learner.Learner(model).learn(data)
            metrics = None
            print('\n*** END OF MODEL TRAINING ***\n')
        elif self.mode == 'inference':
            # 3. Run model inference
            print('*** START OF MODEL PREDICTION ***\n')
            predictions = model.predict(data)
            print('*** END OF MODEL PREDICTION ***\n')
            # # 4. Evaluate the model
            # print\n*** start of model evaluation ***\n')
            # metrics = evaluator.Evaluator(self.model).evaluate(data, predictions)
            # logs = None
            # print('\n*** end of model evaluation ***\n')
        # 5. Visualize the results
        print('*** START OF DASHBOARD VISUALS ***\n')
        dashboard.Dashboard(self).visualize(metrics = metrics, logs = logs)
        print('\n*** END OF DASHBOARD VISUALS ***\n')

    # Run the pipeline
    def run(self):
        print('\n*** START OF PIPELINE ***\n')
        self._build_pipeline()
        print('*** END OF PIPELINE ***\n')
        return None

def main():
    config = {
        'model': 'knn',       # choose from 'null_model', 'knn'
        'dataset': 'forestfires', # choose from 'car', 'breast-cancer-wisconsin', 'house-votes-84', 'abalone', 'machine', 'forestfires', 'racetracks'
        'mode': 'training',          # choose from 'training', 'inference'
        'cross_validation_splits': 1 # number of experiments 'k' to run k x 2 cross validation
    }
    pipeline = Pipeline(config)
    pipeline.run()

if __name__ == '__main__':
    main()


# need a config file that defines the model hyperparameters
# need a config file that defines the dataset to use and how to preprocess it
    

# 1. Load data
# 2. Preprocess data
# 3. Train model
# 4. Evaluate model
# 5. Save model
# 6. Save results
# 7. Save logs