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