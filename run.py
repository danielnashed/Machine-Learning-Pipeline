## Package imports
import os
import sys
from feature_pipeline import data_transformer
from training_pipeline import learner, evaluator, dashboard
from inference_pipeline import model as model_selector


class Pipeline:
    def __init__(self, config):
        self.model, self.data, self.mode = config # filepath

    # Build the pipeline
    def _build_pipeline(self):
        # 1. Load data and preprocess it
        print('\n*** start of data transformation ***\n')
        data = data_transformer.DataTransformer(self.data).process()
        print('\n*** end of data transformation ***\n')
        # 2. Create the model (and train it if in training mode)
        print('\n*** start of model selection ***\n')
        model = model_selector.Model(self.model).select()
        print('\n*** end of model selection ***\n')
        if self.mode == 'training':
            print('\n*** start of model training ***\n')
            model = learner.Learner(model).learn(data)
            print('\n*** end of model training ***\n')
        # 3. Run model inference
        print('\n*** start of model prediction ***\n')
        predictions = model.predict(data)
        print('\n*** end of model prediction ***\n')
        # 4. Evaluate the model
        print('\n*** start of model evaluation ***\n')
        metrics = evaluator.Evaluator(self.model).evaluate(data, predictions)
        print('\n*** end of model evaluation ***\n')
        # 5. Visualize the results
        print('\n*** start of dashboard visuals ***\n')
        dashboard.Dashboard(self.mode).visualize(metrics)
        print('\n*** end of dashboard visuals ***\n')

    # Run the pipeline
    def run(self):
        self._build_pipeline()
        print('Pipeline complete.\n')
        return None

def load_config(model_name, dataset, mode):

    ## Add current working directory to path
    sys.path.append(os.getcwd())

    ## Import the model config file
    model_fullpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'inference_pipeline', model_name)
    
    ## Import the dataset config file
    if dataset == 'car':
        directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'datasets', 'Classification Data Sets', 'Car Evaluation')
    elif dataset == 'breat-cancer-wisconsin':
        directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'datasets', 'Classification Data Sets', 'Breast Cancer')
    elif dataset == 'house-votes-84':
        directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'datasets', 'Classification Data Sets', 'Congressional Voting Records')
    elif dataset == 'abalone':
        directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'datasets', 'Regression Data Sets', 'Abalone')
    elif dataset == 'machine':
        directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'datasets', 'Regression Data Sets', 'Computer Hardware')
    elif dataset == 'forestfires':
        directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'datasets', 'Regression Data Sets', 'Forest Fires')
    elif dataset == 'racetracks':
        directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'datasets', 'Reinforcement Learning Data Sets', 'Racetracks')
    data_fullpath = os.path.join(directory, dataset) 

    return (model_fullpath, data_fullpath, mode)

def main():
    config = load_config(model_name = 'null_model', dataset = 'car', mode = 'training')
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