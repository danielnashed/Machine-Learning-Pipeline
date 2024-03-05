import pandas as pd
import copy
import operator
import numpy as np
import sys
from inference_pipeline.decision_tree import node as TreeNode

# K-Nearest Neighbors (KNN) model
# [description] KNN is a non-parametric, lazy learning algorithm that can be used for classification 
# and regression. It is a type of instance-based learning where training is done at query time. 
# The KNN algorithm assumes that similar instances exist in close proximity. KNN uses the distance metric 
# to find the k-nearest neighbors and then uses a kernel as a function of the distance to classify 
# the query point or predict its value.
#
# [input] is training data and labels
# [output] is trained model and logs
#
class DecisionTree:
    def __init__(self):
        self.hyperparameters = None # k for KNN
        self.prediction_type = None # classification or regression
        self.function = None # function learned from training data
        self.positive_class = None # positive class (if binary classification is used)
        self.num_classes = None # number of classes (if multi-class classification is used)
        self.classes = None # unique classes in the training data
        self.split_criterion = None # split criterion for decision tree
        self.min_samples_leaf = 1 # minimum number of samples in a leaf node (default = 1)
        self.max_depth = float('inf') # maximum depth of the decision tree (default = infinity)
        sys.setrecursionlimit(3000) # set recursion limit to 3000 for large datasets
        
    # Set the hyperparameters for the model (k for KNN)
    def set_params(self, hyperparameters):
        self.hyperparameters = hyperparameters

    # Train the model
    def fit(self, X, y):
        # learning curve: should produce logs of model metric as function of % of trainng data 
        # used for training. There is no learning curve associated with KNN.
        learning_metrics = None
        # deep copy the training data
        data = copy.deepcopy(X)
        # add the target to the data
        data['target'] = y
        # get all unique classes
        self.classes = data['target'].unique()
        # set min_samples_leaf to 10 if the number of samples is greater than 1000 to prevent stack overflow
        if len(data) > 200:
            self.min_samples_leaf = 2
            self.max_depth = 100
        # select split criterion based on either classification or regression
        if self.prediction_type == 'classification':
            self.split_criterion = self.gain_ratio
        elif self.prediction_type == 'regression':
            self.split_criterion = self.squared_error
        # initialize the root node of the decision tree 
        root = TreeNode.Node(data)
        # build the decision tree
        self.function = self.build_tree(root)
        return learning_metrics
    
    def current_entropy(self, data):
        entropy = 0
        for cls in self.classes:
            if cls not in data['target'].unique(): # skip if class is not in data
                continue
            # proportion of examples in dataset that belong to class cls
            p = data['target'].value_counts()[cls]/len(data['target'])
            entropy += -p*np.log2(p)
        return entropy

    def expected_entropy(self, data, feature):
        # if feature is categorical, make a split for each unique value of the feature
        if data[feature].dtype == 'object':
            unique_values = data[feature].unique()
            entropy = 0
            for value in unique_values:
                subset = data[data[feature] == value] # subset of data where feature = value
                entropy += len(subset)/len(data) * self.current_entropy(subset)
        # if feature is continuous, make a binary split at mean of feature
        else:
            mean = data[feature].mean()
            subset1 = data[data[feature] < mean]
            subset2 = data[data[feature] >= mean]
            subset1_entropy = len(subset1)/len(data) * self.current_entropy(subset1)
            subset2_entropy = len(subset2)/len(data) * self.current_entropy(subset2)
            entropy = subset1_entropy + subset2_entropy
        return entropy

    def gain(self, data, feature):
        gain = self.current_entropy(data) - self.expected_entropy(data, feature)
        return gain
    
    def split_info(self, data, feature):
        split_info = 1e-7
        # if feature is categorical, make a split for each unique value of the feature
        if data[feature].dtype == 'object':
            unique_values = data[feature].unique()
            for value in unique_values:
                subset = data[data[feature] == value]
                if len(data) > 0 and len(subset) > 0:
                    split_info += -len(subset)/len(data) * np.log2(len(subset)/len(data))
        # if feature is continuous, make a binary split at mean of feature
        else:
            mean = data[feature].mean()
            subset1 = data[data[feature] < mean]
            subset2 = data[data[feature] >= mean]
            if len(data) > 0 and len(subset1) > 0:
                subset1_split_info = -len(subset1)/len(data) * np.log2(len(subset1)/len(data))
            else:
               subset1_split_info = 1e-7
            if len(data) > 0 and len(subset2) > 0:
                subset2_split_info = -len(subset2)/len(data) * np.log2(len(subset2)/len(data))
            else:
                subset2_split_info = 1e-7
            split_info = subset1_split_info + subset2_split_info
        return split_info

    def gain_ratio(self, data, feature):
        gain_ratio = self.gain(data, feature) / self.split_info(data, feature)
        return gain_ratio
    
    def squared_error(self, data, feature):
        # if feature is categorical, make a split for each unique value of the feature
        if data[feature].dtype == 'object':
            branches = []
            branch_predictions = []
            unique_values = data[feature].unique()
            for value in unique_values:
                subset = data[data[feature] == value]
                branches.append(subset)
                # prediction for each subset is mode of target variable in the subset
                branch_predictions.append(subset['target'].mode()[0])
        # if feature is continuous, make a binary split at mean of feature
        else:
            mean = data[feature].mean()
            subset1 = data[data[feature] < mean]
            subset2 = data[data[feature] >= mean]
            branches = [subset1, subset2]
            # prediction for each subset is mean of target variable in the subset
            branch_predictions = [subset1['target'].mean(), subset2['target'].mean()]
        squared_error = 0
        # iterate over all branches
        for j, branch in enumerate(branches):
            # iterate over all examples in the branch
            for i in range(len(branch)):
                squared_error += (branch['target'].iloc[i] - branch_predictions[j])**2
        return squared_error/len(data)

    def is_base_case(self, node):
        # if the node is pure, then treat as leaf node
        if len(node.data['target'].unique()) == 1:
            return True
        # if node contains less than min_samples_leaf, then treat as leaf node
        if len(node.data) < self.min_samples_leaf:
            return True
        # if depth of node exceeds max_depth, then treat as leaf node
        if node.depth > self.max_depth:
            return True
        return False
        
    def select_best_feature(self, data):
        # dictionary to map operator based on split criterion condition
        ops = {'>': operator.gt, '<': operator.lt}
        op = '>' if self.split_criterion == self.gain_ratio else '<'
        best_feature = None
        # set best_criterion to negative infinity if gain ratio is used, otherwise set to positive infinity
        best_criterion = float('-inf') if self.split_criterion == self.gain_ratio else float('inf')
        # for each feature in the data, calculate the split criterion and select the best feature
        for feature in data.columns[:-1]:
            criterion = self.split_criterion(data, feature)
            if ops[op](criterion, best_criterion): # if criterion is better than best_criterion
                best_criterion = criterion
                best_feature = feature
        return best_feature
    
    def split_into_children(self, node, feature):
        children = []
        # if feature is categorical, make a split for each unique value of the feature
        if node.data[feature].dtype == 'object':
            unique_values = node.data[feature].unique()
            threshold = unique_values
            for value in unique_values:
                subset = node.data[node.data[feature] == value]
                children.append(TreeNode.Node(copy.deepcopy(subset)))
        # if feature is continuous, make a binary split at mean of feature
        else:
            mean = node.data[feature].mean()
            threshold = mean
            subset1 = node.data[node.data[feature] < mean]
            subset2 = node.data[node.data[feature] >= mean]
            children.append(TreeNode.Node(copy.deepcopy(subset1)))
            children.append(TreeNode.Node(copy.deepcopy(subset2)))
        return children, threshold
    
    def build_tree(self, root, depth=0):
        root.depth = depth # set the depth of the node
        # return node if we hit base case
        if self.is_base_case(root):
            return root
        # find the best feature to split on
        best_feature = self.select_best_feature(root.data)
        # split the data in the root into children based on the best feature
        children, threshold = self.split_into_children(root, best_feature)
        root.children = children
        root.feature = best_feature
        root.threshold = threshold
        depth += 1 # increment the depth for the children
        # recursively build the tree for each child
        for child in children:
            self.build_tree(child, depth)
        return root
    
    def traverse_tree(self, query):
        node = self.function
        # keep traversing tree as long as there are children
        while node.children:
            feature = node.feature
            threshold = node.threshold
            # if feature is categorical, traverse to the child node that matches the query value
            if node.data[feature].dtype == 'object':
                value = query[feature]
                indices = np.where(node.threshold == value)[0] # index of the child node
                if indices.size > 0: # if value found in threshold
                    index = indices[0]
                else:
                    index = 0 # if value not found in threshold, go to left child by default
                node = node.children[index]
            # if feature is continuous, traverse to the child node based on the numerical threshold
            else:
                value = query[feature]
                if value < threshold:
                    node = node.children[0] # go to left child if value is less than threshold
                else:
                    node = node.children[1] # go to right child if value otherwise
        # if classification, return the mode of the target variable in the leaf node
        if self.prediction_type == 'classification':
            return node.data['target'].mode()[0]
        # if regression, return the mean of the target variable in the leaf node
        elif self.prediction_type == 'regression':
            return node.data['target'].mean()
    
    # Predict the target values
    def predict(self, X):
        y_pred = []
        # for each query point in X, traverse the decision tree to predict the target value/label
        for i in range(len(X)):
            query = X.iloc[i]
            prediction = self.traverse_tree(query)
            y_pred.append(prediction)
        y_pred = pd.DataFrame(y_pred) # convert to dataframe object
        return y_pred
    


