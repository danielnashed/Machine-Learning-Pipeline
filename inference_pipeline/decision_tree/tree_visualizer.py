from graphviz import Digraph
import numpy as np

class DecisionTreeVisualizer:
    def __init__(self, column_names, prediction_type, output_path):
        self.column_names = column_names
        self.output_path = output_path
        self.prediction_type = prediction_type

    def label_node(self, node):
        # if the node is a leaf, color it and label it with the class man or mode
        if node.children is None:
            # if classification, return the mode of the target variable in the leaf node
            if self.prediction_type == 'classification':
                return str(node.data['target'].mode()[0]), True
            # if regression, return the mean of the target variable in the leaf node
            elif self.prediction_type == 'regression':
                return str(round(node.data['target'].mean(), 2)), True
        # label the node with the feature name
        feature_name = str(node.feature_name)
        # if the feature is not a list or array
        if not isinstance(node.threshold, (list, np.ndarray)):
            # if the feature is a number, round it to 2 decimal places
            if isinstance(node.threshold, (int, float)):
                feature_threshold = str(round(node.threshold, 2))
            else:
                feature_threshold = str(node.threshold)
        # if the feature is a list or numpy array, convert it to a string
        else:
            # replace the forward slash with a backslash for the threshold
            feature_threshold = str([str(x).replace("/", ",").replace(":", ",") for x in node.threshold])
        label = feature_name + '\n' + feature_threshold
        return label, False

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

    def draw_tree(self, root):
        tree = Digraph() # create a new Digraph object
        self.add_edges(tree, root) # add edges to the graph
        tree.render(self.output_path, view=False) # render the graph to a file and display it