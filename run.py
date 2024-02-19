## Package imports
import os
import sys
from datetime import datetime
from feature_pipeline import data_transformer
from training_pipeline import learner, evaluator, dashboard
from inference_pipeline import model as model_selector


# [description] this is the main class that orchestrates the entire end-to-end pipeline to 
# train a model or run inference on a model.
#
# [input] is config dictionary to specify model, dataset, mode, and cross validation splits
# [output] is None
#
class Pipeline:
    def __init__(self, config):
        self.model, self.data, self.mode, self.splits, self.output = self.load_config(config)

    # Process the config dictionary
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

        # Create an output directory to store all exported files during pipeline execution
        directory_name = mode + '_' + model_name + '_' + dataset + '_' + str(datetime.now()).replace(' ', '_').replace(':', '-').split('.')[0]
        output_directory = os.path.join(os.getcwd(), 'output', directory_name)
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        return (model_fullpath, data_fullpath, mode, splits, output_directory)

    # Build the pipeline
    def _build_pipeline(self):
        # 1. Load data and preprocess it
        print('*** START OF DATA TRANSFORMATION ***\n')
        data, pos_class = data_transformer.DataTransformer(self.data, self.mode, self.splits, self.output).process()
        print('\n*** END OF DATA TRANSFORMATION ***\n')
        # 2.a. Create the model and train it
        print('*** START OF MODEL SELECTION ***\n')
        model = model_selector.Model(self.model, pos_class).select()
        print('\n*** END OF MODEL SELECTION ***\n')
        if self.mode == 'training':
            print('*** START OF MODEL TRAINING ***\n')
            model, logs = learner.Learner(model, self.output).learn(data)
            metrics = None
            print('\n*** END OF MODEL TRAINING ***\n')
        elif self.mode == 'inference':
            # 2.b. Run model inference
            print('*** START OF MODEL PREDICTION ***\n')
            predictions = model.predict(data)
            print('*** END OF MODEL PREDICTION ***\n')
        # 3. Visualize the results
        print('*** START OF DASHBOARD VISUALS ***\n')
        dashboard.Dashboard(self).visualize(metrics = metrics, logs = logs)
        print('\n*** END OF DASHBOARD VISUALS ***\n')

    # Run the pipeline
    def run(self):
        print('\n*** START OF PIPELINE ***\n')
        self._build_pipeline()
        print('*** END OF PIPELINE ***\n')

def main():
    config = {
        'model': 'condensed_knn',       # choose from 'null_model', 'knn', 'condensed_knn'
        'dataset': 'abalone', # choose from 'car', 'breast-cancer-wisconsin', 'house-votes-84', 'abalone', 'machine', 'forestfires', 'racetracks'
        'mode': 'training',          # choose from 'training', 'inference'
        'cross_validation_splits': 5 # number of experiments 'k' to run k x 2 cross validation
    }
    pipeline = Pipeline(config)
    pipeline.run()

if __name__ == '__main__':
    main()