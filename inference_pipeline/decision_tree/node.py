
"""
This module contains the Node class which is used to create the decision tree.

The Node class is the building block of the decision tree, where each node in the tree is 
represented as a Node object.

The Node class contains the following attributes:
    - id: the unique identifier of the node
    - data: the data in the node
    - feature: the feature to split on
    - feature_name: the name of the feature
    - threshold: the threshold to split on
    - children: the children of the node
    - grandparent: the grandparent of the node
    - visited: a flag to indicate if the node has been visited during pruning
    - depth: the depth of the node in the tree

The Node class contains the following methods:
    - None
"""

class Node:
    def __init__(self, data, id=0, feature=None, threshold=None, children=None):
        self.id = id
        self.data = data
        self.feature = feature
        self.feature_name = None
        self.threshold = threshold
        self.children = children
        self.grandparent = None
        self.visited = False
        self.depth = 0
