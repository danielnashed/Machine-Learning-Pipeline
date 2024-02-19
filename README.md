# EN.605.611 - Introduction to Machine Learning
### Author: Daniel Nashed

## Package Description
This package is an end-to-end machine learing pipeline to be used to develop, train and run inference of machine learning models.

## Package File Structure

📦Machine Learning Pipeline<br>
 ┃ <br>
 ┣ 📂datasets<br>
 ┃ ┃ <br>
 ┃ ┣ 📂Classification Data Sets<br>
 ┃ ┃ ┃ <br>
 ┃ ┃ ┣ 📂Breast Cancer<br>
 ┃ ┃ ┃ ┣ breast-cancer-wisconsin.config --> structured INI file for processing dataset<br> 
 ┃ ┃ ┃ ┣ breast-cancer-wisconsin.names --> unstructured txt file with dataset description<br>
 ┃ ┃ ┃ ┗ breast-cancer-wisconsin.data --> txt file containing data<br>
 ┃ ┃ ┃ <br>
 ┃ ┃ ┣ 📂Car Evaluation<br>
 ┃ ┃ ┃ ┣ car.config --> structured INI file for processing dataset<br> 
 ┃ ┃ ┃ ┣ car.names --> unstructured txt file with dataset description<br>
 ┃ ┃ ┃ ┗ car.data --> txt file containing data<br>
 ┃ ┃ ┃ <br>
 ┃ ┃ ┗ 📂Congressional Vote<br>
 ┃ ┃ ┃ ┣ house-votes-84.config --> structured INI file for processing dataset<br> 
 ┃ ┃ ┃ ┣ house-votes-84.names --> unstructured txt file with dataset description<br>
 ┃ ┃ ┃ ┗ house-votes-84.data --> txt file containing data<br>
 ┃ ┃ <br>
 ┃ ┣ 📂Regression Data Sets<br>
 ┃ ┃ ┃ <br>
 ┃ ┃ ┣ 📂Abalone<br>
 ┃ ┃ ┃ ┣ abalone.config  --> structured INI file for processing dataset<br>
 ┃ ┃ ┃ ┣ abalone.names --> unstructured txt file with dataset description<br>
 ┃ ┃ ┃ ┗ abalone.data --> txt file containing data<br>
 ┃ ┃ ┃ <br>
 ┃ ┃ ┣ 📂Computer Hardware<br>
 ┃ ┃ ┃ ┣ machine.config --> structured INI file for processing dataset<br> 
 ┃ ┃ ┃ ┣ machine.names --> unstructured txt file with dataset description<br>
 ┃ ┃ ┃ ┗ machine.data --> txt file containing data<br>
 ┃ ┃ ┃ <br>
 ┃ ┃ ┗ 📂Forest Fires<br>
 ┃ ┃ ┃ ┣ forestfires.config --> structured INI file for processing dataset<br> 
 ┃ ┃ ┃ ┣ forestfires.names --> unstructured txt file with dataset description<br>
 ┃ ┃ ┃ ┗ forestfires.data --> txt file containing data<br>
 ┃ ┃ ┃ <br>
 ┃ ┣ 📂Reinforcement Learning Data Sets<br>
 ┃ ┃ ┃ <br>
 ┃ ┃ ┗ 📂Racetracks<br>
 ┃ ┃ ┃ ┣ L-track.txt --> txt file containing data<br>
 ┃ ┃ ┃ ┣ O-track.txt --> txt file containing data<br>
 ┃ ┃ ┃ ┣ R-track.txt --> txt file containing data<br>
 ┃ ┃ ┃ ┗ W-track.txt --> txt file containing data<br>
 ┃ ┃ <br>
 ┃ ┗ template.config<br>
 ┃<br>
 ┣ 📂feature_pipeline<br>
 ┃ ┗ data_transformer.py --> class with methods for pre-processing of dataset<br>
 ┃ <br>
 ┣ 📂training_pipeline<br>
 ┃ ┣ learner.py --> class with methods to train an ml model<br>
 ┃ ┣ evaluator.py --> class with methods to evaluate model performance<br>
 ┃ ┗ dashboard.py --> class with methods to display evaluation metrics<br>
 ┃ <br>
 ┣ 📂inference_pipeline <br>
 ┃ ┃ <br>
 ┃ ┣ model.py --> abstract class interface to represent an ml model<br>
 ┃ ┃ <br>
 ┃ ┣ 📂null_model --> baseline model using mean or mode of target class<br>
 ┃ ┃ ┣ null_model.py --> class and methods to model a null model<br>
 ┃ ┃ ┗ null_model.config --> structured INI file to configure model<br>
 ┃ ┃ <br>
 ┃ ┣ 📂knn --> k-nearest neighbour model<br>
 ┃ ┃ ┣ knn.py --> class and methods to model a knn model<br>
 ┃ ┃ ┗ knn.config --> structured INI file to configure model<br>
 ┃ ┃ <br>
 ┃ ┣ 📂edited_knn --> edited k-nearest neighbour model to improve inference time<br>
 ┃ ┃ ┣ edited_knn.py --> class and methods to model a knn model<br>
 ┃ ┃ ┗ edited_knn.config --> structured INI file to configure model<<br>
 ┃ <br>
 ┣ 📂output --> directory to hold outputs of pipelines<br>
 ┃ ┃ <br>
 ┃ ┣ run.py --> entry point for package pipeline<br>
 ┃ ┗ README.md<br>
 ┃<br>

## Installation & Running Instructions
### Cloning the Package
Clone the root directory named "Machine Learning Pipeline"

### Python
The project has been written using Python 3.12.1

### Package Requirements
All dependencies required for the project have been listed in the file `requirements.txt`. Use the following command to install all dependencies at once: 
`pip3 install -r requirements.txt` 

### Running the Package
The entry point is provided in run.py located in the root directory. To run a pipeline, modify the config dictionary inside the main method as follows:

- 'model': choose from 'null_model', 'knn', 'edited_knn'
- 'dataset': choose from 'car', 'breast-cancer-wisconsin' 'house-votes-84', 'abalone',  'machine', 'forestfires', 'racetracks'
- 'mode': choose from 'training', 'inference'
- 'cross_validation_splits': number of experiments 'k' to run k x 2 cross validation

### Modifying Config Files
Each model and each datset has its own structured INI config file that contains configuturable parameters which can be modified by users. 

For example, for datasets, the choice of what missing values to impute or ignore, whether a categorical feature is ordinal or nominal, what type of transformation to apply (standardization or normalization), which class label is the positive class, and so on can all be specified within the respective config file of each dataset. 

For models, all hyperparameters can specified in their respective config files as well as a key-value pair where the key is the name of the hyperparameter and the value is a list of values for the hyperparameter. If no hypertuning is required, then the value is simply a list of size 1. The type of prediction to perform, either regression, binary-classification or multi-classification can also be specified. In addition, the choice of which metrics to use for evaluation can be specified in the config file. If multiple metrics are needed for evaluation, simply set the value of the respective metric key to 1. Otherwise, keep value at 0 if you do not wish to use the metric for evaluation.


### Outputs
Each time a pipeline is executed, a new subdirectory is created inside 'outputs' directory to hold all exported data pertaining to an executed pipeline. The following files are exported during pipeline execution:

- pickle file containing the trained model 
- pickle file containing logs for the model during hyperparameter tuning & training
- csv file containing processed train dataset prior to transformation and k-fold splits
- csv file containing processed validation dataset prior to transformation and k-fold splits
- png image for learning curves of the model during training 
- png image for validation curves of the model during training