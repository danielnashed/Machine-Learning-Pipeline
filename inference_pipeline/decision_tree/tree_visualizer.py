from graphviz import Digraph
import numpy as np

"""
This module contains the DecisionTreeVisualizer class which is used to visualize the decision tree
using the graphviz library. It exports the decision tree as a pdf file.

The DecisionTreeVisualizer class contains the following attributes:
    - column_names: list of column names
    - output_path: path to save the decision tree visualization
    - prediction_type: 'classification' or 'regression'

The DecisionTreeVisualizer class contains the following methods:
    - label_node: label the node with the feature name and threshold
    - add_edges: add edges to the graph for each child node
    - draw_tree: draw the decision tree using the graphviz library
"""
class DecisionTreeVisualizer:
    def __init__(self, column_names, prediction_type, output_path):
        self.column_names = column_names
        self.output_path = output_path
        self.prediction_type = prediction_type
    
    """
    'label_node' method is responsible for labeling the node with the feature name and threshold.
    Args:
        node (Node object): the node to label
    Returns:
        label (string): label for the node
        is_leaf (boolean): True if the node is a leaf, False otherwise
    """
    def label_node(self, node):
        # leaves have no children
        if node.children is None:
            # if classification, return the mode of the target variable in the leaf node
            if self.prediction_type == 'classification':
                return str(node.data['target'].mode()[0]), True
            # if regression, return the mean of the target variable in the leaf node
            elif self.prediction_type == 'regression':
                return str(round(node.data['target'].mean(), 2)), True
        feature_name = str(node.feature_name) # label node with feature name
        # if the feature is not a list or array
        if not isinstance(node.threshold, (list, np.ndarray)):
            # if the feature is a number, round it to 2 decimal places
            if isinstance(node.threshold, (int, float)):
                feature_threshold = str(round(node.threshold, 2))
            else:
                feature_threshold = str(node.threshold)
        # if the feature is a list or numpy array, convert it to a string
        else:
            # replace delimiters forbidden in graphviz labels
            feature_threshold = str([str(x).replace("/", ",").replace(":", ",") for x in node.threshold])
        label = feature_name + '\n' + feature_threshold
        return label, False

    """
    'add_edges' method is responsible for adding edges to the graph for each child node.
    Args:
        tree (Digraph object): the graph to add edges to
        node (Node object): the node to add edges for
    Returns:
        None
    """
    def add_edges(self, tree, node):
        if node is None:
            return
        # recursively add edges to the graph for each child
        if node.children:
            node_data, _ = self.label_node(node) # create a label for the node
            node_id = f"{node_data}_{id(node)}" # create a unique id for the node
            tree.node(node_id, label=node_data) # add the node to the graph
            for child in node.children:
                child_data, is_leaf = self.label_node(child)
                child_id = f"{child_data}_{id(child)}"
                # if the node is a leaf, color it and label it with the class
                if is_leaf:
                    tree.node(child_id, label=child_data, style='filled', fillcolor='lightblue')
                else:
                    tree.node(child_id, label=child_data)
                tree.edge(node_id, child_id) # add an edge from the parent to the child
                self.add_edges(tree, child) # enter recursion

    """
    'draw_tree' method is responsible for drawing the decision tree using the graphviz library.
    Args:
        root (Node object): the root node of the decision tree
    Returns:
        None
    """
    def draw_tree(self, root):
        tree = Digraph() # create a new Digraph object
        self.add_edges(tree, root) # add edges to the graph
        tree.render(self.output_path, view=False) # render the graph and save to filepath