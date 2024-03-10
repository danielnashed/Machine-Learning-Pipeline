import pandas as pd
import copy
import operator
import numpy as np
import sys
from inference_pipeline.decision_tree import node as TreeNode
from inference_pipeline.decision_tree import tree_pruner as TreePruner
from inference_pipeline.decision_tree import tree_visualizer as TreeVisualizer

"""
This module contains the DecisionTree class which is used to create a decision tree for training
and predicting target values. The decision tree is built using the ID3 algorithm and can be used
for both classification and regression tasks.

The DecisionTree class contains the following attributes:
    - id: id of last node in the decision tree
    - list_of_nodes: list of node id's in the decision tree
    - non_grandparents: list of non-grandparent node id's in the decision tree
    - hyperparameters: k for KNN
    - prediction_type: 'classification' or 'regression'
    - function: function learned from training data
    - positive_class: positive class (if binary classification is used)
    - num_classes: number of classes (if multi-class classification is used)
    - column_names: column names for the dataset
    - classes: unique classes in the training data
    - split_criterion: split criterion for decision tree
    - min_samples_leaf: minimum number of samples in a leaf node (default = 1)
    - max_depth: maximum depth of the decision tree (default = infinity)
    - pruning: whether to perform pruning on the decision tree
    - validation_set: validation set for pruning

The DecisionTree class contains the following methods:
    - set_params: set the hyperparameters for the model
    - fit: train the model
    - optimal_split_value: find the optimal split value for a continuous feature
    - current_entropy: calculate the current entropy of the dataset
    - expected_entropy: calculate the expected entropy of the dataset
    - gain: calculate the gain of the dataset
    - split_info: calculate the split info of the dataset
    - gain_ratio: calculate the gain ratio of the dataset
    - squared_error: calculate the squared error of the dataset
    - is_base_case: check if the node is a base case
    - select_best_feature: select the best feature to split on
    - split_into_children: split the data into children based on the best feature
    - is_grandparent: check if the node is a grandparent
    - build_tree: build the decision tree
    - traverse_tree: traverse the decision tree to predict the target values
    - predict: predict the target values
"""

class DecisionTree:
    def __init__(self):
        self.id = 0 # id of nodes in the decision tree
        self.list_of_nodes = {} # list of node id's in the decision tree
        self.non_grandparents = [] # list of non-grandparent node id's in the decision tree
        self.hyperparameters = None # k for KNN
        self.prediction_type = None # classification or regression
        self.function = None # function learned from training data
        self.positive_class = None # positive class (if binary classification is used)
        self.num_classes = None # number of classes (if multi-class classification is used)
        self.column_names = None # column names for the dataset
        self.classes = None # unique classes in the training data
        self.split_criterion = None # split criterion for decision tree
        self.min_samples_leaf = 1 # minimum number of samples in a leaf node (default = 1)
        self.max_depth = float('inf') # maximum depth of the decision tree (default = infinity)
        self.pruning = False # whether to perform pruning on the decision tree
        self.validation_set = None # validation set for pruning
        # dictionary to map operator based on split criterion condition
        self.ops = {'>': operator.gt, '<': operator.lt}
        self.op = None # operator based on split criterion condition
        sys.setrecursionlimit(3000) # set recursion limit to 3000 for large datasets

    """
    'set_params' method is responsible for setting the hyperparameters for the model.
    Args:
        hyperparameters (dictionary): hyperparameters for the model
    Returns:
        None
    """
    def set_params(self, hyperparameters):
        self.hyperparameters = hyperparameters

    """
    'fit' method is responsible for training the model.
    Args:
        X (DataFrame): training data
        y (DataFrame): target variable
    Returns:
        learning_metrics (dictionary): logs of model metric as function of % of training data
    """
    def fit(self, X, y):
        # learning curve: should produce logs of model metric as function of % of training data 
        # used for training. No learning curve is produced for decision tree.
        learning_metrics = None
        data = copy.deepcopy(X) # deep copy training data
        data['target'] = y # add target to data
        self.classes = data['target'].unique() # get all unique classes in the target variable
        # select split criterion based on either classification or regression
        if self.prediction_type == 'classification':
            self.split_criterion = self.gain_ratio
            self.op = '>'
        elif self.prediction_type == 'regression':
            self.split_criterion = self.squared_error
            self.op = '<'
        # initialize the root node of the decision tree 
        root = TreeNode.Node(data, id=self.id)
        # add root node id to list of nodes
        self.list_of_nodes[0] = [[root.id]]
        # build the decision tree
        self.function = self.build_tree(root)
        # prune the trained decision tree if pruning is set to True
        if self.pruning:
            self.function = TreePruner.DecisionTreePruner(self).prune_tree()
        return learning_metrics
    
    """
    'optimal_split_value' method is responsible for finding the optimal split value for a 
    continuous feature. The optimal split value is the value that maximizes the gain ratio for 
    classification or minimizes the squared error for regression. To improve the efficiency of
    the algorithm, the method uses the following approach:
        - Sort the data by the feature
        - Iterate over the data and compute the gain ratio at each split point where the 
          class label changes (for classification). For regression, discretize the feature into 
          4 bins and compute the squared error at each split point between the bins. 
        - Return the split point that maximizes the gain ratio or minimizes the squared error.
    Args:
        data (DataFrame): training data
        feature (string): feature to split on
    Returns:
        best_split_value (float): optimal split value for the feature
    """
    def optimal_split_value(self, data, feature):
        data = copy.deepcopy(data) # create a copy of the data 
        data = data.sort_values(feature) # sort the data by the feature
        best_split_value = None # initialize best split value
        # if classification, iterate over all data and compute the gain ratio at each split point where class label changes
        if self.prediction_type == 'classification':
            best_gain_ratio = float('-inf')
            class_label = data['target'].iloc[0] # get first class label in dataset
            for i in range(len(data)):
                # if class label changes
                if data['target'].iloc[i] != class_label:
                    class_label = data['target'].iloc[i] # update class label
                    split_value = (data[feature].iloc[i] + data[feature].iloc[i-1])/2
                    gain_ratio = self.gain_ratio(data, feature, split_value)
                    if gain_ratio > best_gain_ratio:
                        best_gain_ratio = gain_ratio
                        best_split_value = split_value
            # if class label never changed, then select mean of feature by default
            if best_split_value == None:
                best_split_value = data[feature].mean()
        # if regression, discretize the feature into 4 bins and compute the squared error at each split point between the bins
        elif self.prediction_type == 'regression':
            best_mse = float('inf')
            # discretize the feature into 4 bins
            #bins = np.array_split(data[feature], 4)
            bins = pd.cut(data[feature], bins=4)
            for i in range(3):
                left = bins.cat.categories[i].right # get the right edge of the left bin
                right = bins.cat.categories[i+1].left # get the left edge of the right bin
                split_value = (left + right)/2
                #split_value = (bins[i].max() + bins[i+1].min())/2
                mse = self.squared_error(data, feature, split_value)
                if mse < best_mse:
                    best_mse = mse
                    best_split_value = split_value
        return best_split_value

    """
    'current_entropy' method is responsible for calculating the current entropy of the dataset.
    This is a measure of the impurity of the dataset. The entropy is calculated using the formula:
        entropy = -p1*log2(p1) - p2*log2(p2) - ... - pn*log2(pn)
    Args:
        data (DataFrame): training data
    Returns:
        entropy (float): current entropy of the dataset
    """
    def current_entropy(self, data):
        entropy = 0
        for cls in self.classes:
            if cls not in data['target'].unique(): # skip if class is not in data
                continue
            # proportion of examples in dataset that belong to class cls
            p = data['target'].value_counts()[cls]/len(data['target'])
            entropy += -p*np.log2(p)
        return entropy

    """
    'expected_entropy' method is responsible for calculating the expected entropy of the dataset.
    This is a measure of the expected impurity of the dataset after a split. The expected entropy
    is calculated using the formula:
        entropy = p1*entropy(subset1) + p2*entropy(subset2) + ... + pn*entropy(subsetn)
    Args:
        data (DataFrame): training data
        feature (string): feature to split on
        split_value (float): split value for continuous feature
    Returns:
        entropy (float): expected entropy of the dataset
    """
    def expected_entropy(self, data, feature, split_value):
        # if feature is categorical, make a split for each unique value of the feature
        if data[feature].dtype == 'object':
            unique_values = data[feature].unique()
            entropy = 0
            for value in unique_values:
                subset = data[data[feature] == value] # subset of data where feature = value
                entropy += len(subset)/len(data) * self.current_entropy(subset)
        # if feature is continuous, make a binary split at optimal value of feature
        else:
            subset1 = data[data[feature] < split_value]
            subset2 = data[data[feature] >= split_value]
            subset1_entropy = len(subset1)/len(data) * self.current_entropy(subset1)
            subset2_entropy = len(subset2)/len(data) * self.current_entropy(subset2)
            entropy = subset1_entropy + subset2_entropy
        return entropy

    """
    'gain' method is responsible for calculating the gain of the dataset. The gain is a measure of
    the reduction in entropy after a split. The gain is calculated using the formula:
        gain = current_entropy - expected_entropy
    Args:
        data (DataFrame): training data
        feature (string): feature to split on
        split_value (float): split value for continuous feature
    Returns:
        gain (float): gain of the dataset
    """
    def gain(self, data, feature, split_value):
        gain = self.current_entropy(data) - self.expected_entropy(data, feature, split_value)
        return gain
    
    """
    'split_info' method is responsible for calculating the split info of the dataset. The split info
    is a measure of the amount of information required to specify the split. The split info is 
    calculated using the formula:
        split_info = -p1*log2(p1) - p2*log2(p2) - ... - pn*log2(pn)
    Args:
        data (DataFrame): training data
        feature (string): feature to split on
        split_value (float): split value for continuous feature
    Returns:
        split_info (float): split info of the dataset
    """
    def split_info(self, data, feature, split_value):
        split_info = 1e-7 # so we dont divide by zero
        # if feature is categorical, make a split for each unique value of the feature
        if data[feature].dtype == 'object':
            unique_values = data[feature].unique()
            for value in unique_values:
                subset = data[data[feature] == value]
                if len(data) > 0 and len(subset) > 0:
                    split_info += -len(subset)/len(data) * np.log2(len(subset)/len(data))
        # if feature is continuous, make a binary split at optimal value of feature
        else:
            # get the two subsets of the data
            subset1 = data[data[feature] < split_value]
            subset2 = data[data[feature] >= split_value]
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

    """
    'gain_ratio' method is responsible for calculating the gain ratio of the dataset. The gain ratio
    is a measure of reduction in entropy after splitting, normalized by the amount of information
    required to specify the split. The gain ratio is calculated using the formula:
        gain_ratio = gain / split_info
    Args:
        data (DataFrame): training data
        feature (string): feature to split on
        split_value (float): split value for continuous feature
    Returns:
        gain_ratio (float): gain ratio of the dataset
    """
    def gain_ratio(self, data, feature, split_value):
        # if all values of the feature are the same, then return very small gain ratio
        if len(data[feature].unique()) == 1:
            return float('-inf')
        # if all values of feature lie on same side of split value, then return very small gain ratio
        if split_value is not None and all(data[feature].unique() >= split_value):
            return float('-inf')
        gain_ratio = self.gain(data, feature, split_value) / self.split_info(data, feature, split_value)
        return gain_ratio
    
    """
    'squared_error' method is responsible for calculating the squared error of the dataset. The 
    squared error is a measure of the average squared difference between the actual and predicted 
    values. The squared error is calculated using the formula:
        squared_error = (actual - predicted)^2
    Args:
        data (DataFrame): training data
        feature (string): feature to split on
    Returns:
        squared_error (float): squared error of the dataset
    """
    def squared_error(self, data, feature, split_value):
        # if all values of the feature are the same, then return very large error
        if len(data[feature].unique()) == 1:
            return float('inf')
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
        # if feature is continuous, make a binary split at optimal value of feature
        else:
            subset1 = data[data[feature] < split_value]
            subset2 = data[data[feature] >= split_value]
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

    """
    'is_base_case' method is responsible for checking if the node is a base case. A node is a base
    case if it is pure, contains one feature, contains less than min_samples_leaf examples, or has
    a depth greater than max_depth.
    Args:
        node (Node object): the node to check
    Returns:
        flag (boolean): True if the node is a base case, False otherwise
    """
    def is_base_case(self, node):
        if self.prediction_type == 'classification':
            # if the node is pure, then treat as leaf node
            flag1 = len(node.data['target'].unique()) == 1
        elif self.prediction_type == 'regression':
            # if the node is pure, then treat as leaf node
            flag1 = (node.data['target'].max() -  node.data['target'].min()) < 1
        # if node contains one feature, then treat as leaf node
        flag2 = len(node.data.columns) == 1
        # if node contains less than min_samples_leaf, then treat as leaf node
        flag3 = len(node.data) < self.min_samples_leaf
        # if depth of node exceeds max_depth, then treat as leaf node
        flag4 = node.depth > self.max_depth
        # if any of the conditions are met, then treat as leaf node
        if flag1 or flag2 or flag3 or flag4:
            node.grandparent = False
            return True
        return False
        
    """
    'select_best_feature' method is responsible for selecting the best feature to split on. The best
    feature is the feature that maximizes the gain ratio for classification or minimizes the squared
    error for regression. The algorithm iterates over all features, calculates the split criterion 
    for each feature, and selects the feature with the best split criterion.
    Args:   
        data (DataFrame): training data
    Returns:
        best_feature (string): best feature to split on
    """
    def select_best_feature(self, data):
        best_feature = None
        best_split_value = None
        # set best_criterion to negative infinity if gain ratio is used, otherwise set to positive infinity
        best_criterion = float('-inf') if self.split_criterion == self.gain_ratio else float('inf')
        # for each feature in the data, calculate the split criterion and select the best feature
        for feature in data.columns[:-1]:
            # if feature is continous, determine the optimal binary split value
            if data[feature].dtype != 'object':
                split_value = self.optimal_split_value(data, feature)
                #split_value = data[feature].mean()
            else:
                split_value = None
            criterion = self.split_criterion(data, feature, split_value)
            if self.ops[self.op](criterion, best_criterion): # if criterion is better than best_criterion
                best_criterion = criterion
                best_feature = feature
                best_split_value = split_value
        return (best_feature, best_split_value)
    
    """
    'split_into_children' method is responsible for splitting the data into children based on the best
    feature. The method splits the data into children based on the unique values of the feature for
    categorical features, or based on the optimal split point of the feature for continuous features. 
    The method creates a child node for each subset of the data and stores the data subset in the child.
    Args:
        node (Node object): the node to split
        feature (string): the feature to split on
    Returns:
        children (list): list of children nodes
        children_ids (list): list of children node ids
        threshold (float): threshold for the feature
    """
    def split_into_children(self, node, feature, split_value):
        children = []
        children_ids = []
        # if feature is categorical, make a split for each unique value of the feature
        if node.data[feature].dtype == 'object':
            unique_values = node.data[feature].unique()
            threshold = unique_values
            for value in unique_values:
                self.id += 1 # increment the id for the child node
                children_ids.append(self.id)
                subset = node.data[node.data[feature] == value]
                subset = subset.drop(feature, axis=1) # drop the feature from the subset
                children.append(TreeNode.Node(copy.deepcopy(subset), id=self.id))
        # if feature is continuous, make a binary split at optimal value of feature
        else:
            threshold = split_value
            subset1 = node.data[node.data[feature] < split_value]
            subset1 = subset1.drop(feature, axis=1) # drop the feature from the subset
            subset2 = node.data[node.data[feature] >= split_value]
            subset2 = subset2.drop(feature, axis=1) # drop the feature from the subset
            if subset1.empty or subset2.empty:
                debug = ''
            self.id += 1
            children_ids.append(self.id)
            children.append(TreeNode.Node(copy.deepcopy(subset1), id=self.id))
            self.id += 1
            children_ids.append(self.id)
            children.append(TreeNode.Node(copy.deepcopy(subset2), id=self.id))
        return children, children_ids, threshold
    
    """
    'is_grandparent' method is responsible for checking if the node is a grandparent. A node is a
    grandparent if it has at least one child that is not a leaf. The method adds the node to the
    list of non-grandparents if all children are leaves.
    Args:
        node (Node object): the node to check
        children (list): list of children nodes
    Returns:
        flag (boolean): True if the node is a grandparent, False otherwise
    """
    def is_grandparent(self, node, children):
        # for node to be a grandparent, it must have at least one child that is not a leaf
        for child in children:
            if child.children:
                return True
        # if all children are leaves, then node is not a grandparent
        self.non_grandparents.append(node.id)
        return False

    """
    'build_tree' method is responsible for building the decision tree. The method recursively builds
    the tree by selecting the best feature to split on, splitting the data into children based on the
    best feature, and building the tree for each child.
    Args:
        root (Node object): the root node of the tree
        depth (int): the depth of the node
    Returns:
        root (Node object): the root node of the tree
    """
    def build_tree(self, root, depth=0):
        root.depth = depth # set the depth of the node
        # return node if we hit base case
        if self.is_base_case(root):
            return root
        # find the best feature to split on
        best_feature, best_split_value = self.select_best_feature(root.data)
        # if no best feature found, then return node
        if best_feature is None:
            return root
        # split the data in the root into children based on the best feature
        children, children_ids, threshold = self.split_into_children(root, best_feature, best_split_value)
        root.children = children
        root.feature = best_feature
        root.feature_name = self.column_names[best_feature]
        root.threshold = threshold
        depth += 1 # increment the depth for the children
        # add children ids to list of nodes
        if depth not in self.list_of_nodes:
            self.list_of_nodes[depth] = []
        self.list_of_nodes[depth].append(children_ids)
        # recursively build the tree for each child
        for child in children:
            self.build_tree(child, depth)
        # After visiting all children of the root, tag root as grandparent or not
        root.grandparent = self.is_grandparent(root, children)
        return root
    
    """
    'traverse_tree' method is responsible for traversing the decision tree to predict the target
    values. The method traverses the tree by starting at the root and following the path of the 
    query point down the branches until it reaches a leaf. The method returns the mode of the target
    variable in the leaf node for classification, or the mean of the target variable in the leaf node.
    Args:
        root (Node object): the root node of the tree
        query (DataFrame): query point
    Returns:
        prediction (float): predicted target value/label
    """
    def traverse_tree(self, root, query):
        node = root
        # keep traversing tree as long as there are children
        while node.children:
            feature = node.feature
            threshold = node.threshold
            # if feature is categorical, traverse to the child node that matches the query value
            if node.data[feature].dtype == 'object':
                value = query[feature]
                # index of the child node that matches the query value
                indices = np.where(node.threshold == value)[0]
                if indices.size > 0: # if value found in threshold
                    index = indices[0]
                # if value not found in threshold, choose a child node at random.
                else:
                    index = np.random.choice(len(node.children))
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
    
    """
    'predict' method is responsible for predicting the target values. The method traverses the decision
    tree to predict the target values for the query points. The method returns the predicted target
    values as a DataFrame object.
    Args:
        X (DataFrame): query points
        function (Node object): function learned from training data (decision tree in this case)
    Returns:
        y_pred (DataFrame): predicted target values
    """
    def predict(self, X, function=None):
        if function:
            root = function # if a function is provided, use it to predict
        else:
            root = self.function # otherwise, use the trained function to predict
        y_pred = []
        # for each query point in X, traverse the decision tree to predict the target value/label
        for i in range(len(X)):
            query = X.iloc[i]
            prediction = self.traverse_tree(root, query)
            y_pred.append(prediction)
        y_pred = pd.DataFrame(y_pred) # convert to dataframe object
        return y_pred


