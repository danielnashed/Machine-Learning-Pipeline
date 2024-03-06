class Node:
    def __init__(self, data, id=0, feature=None, threshold=None, children=None):
        self.id = id
        self.data = data
        self.feature = feature
        self.threshold = threshold
        self.children = children
        self.grandparent = None
        self.depth = 0