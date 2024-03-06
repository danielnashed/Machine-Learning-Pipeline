class Node:
    id_counter = 0  # class variable to keep track of the last assigned node id
    def __init__(self, data, feature=None, threshold=None, children=None):
        self.id = Node.id_counter  # assign the current node id
        Node.id_counter += 1 # increment the node id counter
        self.data = data
        self.feature = feature
        self.threshold = threshold
        self.children = children
        self.depth = 0
        # self.visited = False
        # self.parent = None