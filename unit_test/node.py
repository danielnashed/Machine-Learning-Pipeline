class Node:
    def __init__(self, data, id=0, feature=None, threshold=None, children=None):
        self.id = id
        self.data = data
        self.feature = feature
        self.threshold = threshold
        self.children = children
        self.depth = 0
        self.grandparent = True
        # self.left = left
        # self.right = right

    # def predict(self, sample):
    #     if sample[self.feature] <= self.threshold:
    #         return self.left.predict(sample)
    #     else:
    #         return self.right.predict(sample)