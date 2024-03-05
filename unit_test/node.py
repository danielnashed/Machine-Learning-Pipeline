class Node:
    def __init__(self, data, feature=None, threshold=None, children=None):
        self.data = data
        self.feature = feature
        self.threshold = threshold
        self.children = children
        self.depth = 0