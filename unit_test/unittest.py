import pandas as pd
import copy
import operator
import numpy as np
import node as TreeNode

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
        # select split criterion based on either classification or regression
        if self.prediction_type == 'classification':
            self.split_criterion = self.gain_ratio
            #self.function = data
        elif self.prediction_type == 'regression':
            #self.function = data
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
        split_info = 0
        # if feature is categorical, make a split for each unique value of the feature
        if data[feature].dtype == 'object':
            unique_values = data[feature].unique()
            for value in unique_values:
                subset = data[data[feature] == value]
                split_info += -len(subset)/len(data) * np.log2(len(subset)/len(data))
        # if feature is continuous, make a binary split at mean of feature
        else:
            mean = data[feature].mean()
            subset1 = data[data[feature] < mean]
            subset2 = data[data[feature] >= mean]
            subset1_split_info = -len(subset1)/len(data) * np.log2(len(subset1)/len(data))
            subset2_split_info = -len(subset2)/len(data) * np.log2(len(subset2)/len(data))
            split_info = subset1_split_info + subset2_split_info
        return split_info

    def gain_ratio(self, data, feature):
        gain_ratio = self.gain(data, feature) / self.split_info(data, feature)
        return gain_ratio
    
    def squared_error(self):
        pass

    def is_base_case(self, node):
        # if the node is pure, return the node
        if len(node.data['target'].unique()) == 1:
            return node
        # # if there are no features left to split on, return the node
        # if len(node.data.columns) == 1:
        #     return node
        # if there is only one data example left, return the node
        if len(node.data) == 1:
            return node
        
    def select_best_feature(self, data):
        # dictionary to map operator based on split criterion condition
        ops = {'>': operator.gt, '<': operator.lt}
        op = '>' if self.split_criterion == self.gain_ratio else '<'
        best_feature = None
        best_criterion = 0
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
    
    def build_tree(self, root):
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
        # recursively build the tree for each child
        for child in children:
            self.build_tree(child)
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
                index = np.where(node.threshold == value)[0][0] # index of the child node
                node = node.children[index]
            # if feature is continuous, traverse to the child node based on the numerical threshold
            else:
                value = query[feature]
                if value < threshold:
                    node = node.children[0] # go to left child if value is less than threshold
                else:
                    node = node.children[1] # go to right child if value otherwise
        return node.data['target'].mode()[0] # return the mode of the target variable in the leaf node
    
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


def main():
    # Test the DecisionTree class
    dt = DecisionTree()
    dt.prediction_type = 'classification'
    dt.set_params(None)
    data = [['sunny', 'hot', 'high', 'false', 'no'],
            ['sunny', 'hot', 'high', 'true', 'no'],
            ['overcast', 'hot', 'high', 'false', 'yes'],
            ['rain', 'mild', 'high', 'false', 'yes'],
            ['rain', 'cool', 'normal', 'false', 'yes'],
            ['rain', 'cool', 'normal', 'true', 'no'],
            ['overcast', 'cool', 'normal', 'true', 'yes'],
            ['sunny', 'mild', 'high', 'false', 'no'],
            ['sunny', 'cool', 'normal', 'false', 'yes'],
            ['rain', 'mild', 'normal', 'false', 'yes'],
            ['sunny', 'mild', 'normal', 'true', 'yes'],
            ['overcast', 'mild', 'high', 'true', 'yes'],
            ['overcast', 'hot', 'normal', 'false', 'yes'],
            ['rain', 'mild', 'high', 'true', 'no']]
    data = pd.DataFrame(data, columns=['outlook', 'temperature', 'humidity', 'windy', 'play'])
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    dt.fit(X, y)
    print(dt.function)
    test_X = [['overcast', 'hot', 'high', 'false'],
        ['sunny', 'hot', 'high', 'false'],
            ['sunny', 'hot', 'high', 'true']]
    test_X = pd.DataFrame(test_X, columns=['outlook', 'temperature', 'humidity', 'windy'])
    predictions = dt.predict(test_X)
    print(predictions)

if __name__ == "__main__":
    main()

