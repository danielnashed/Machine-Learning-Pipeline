
class MetaData():
    def __init__(self):
        self.data = None
        self.meta = {}

    def set_meta(self, data, column_names=None, positive_class=None, num_classes=None):
        self.data = data
        self.meta['column_names'] = column_names
        self.meta['pos_class'] = positive_class
        self.meta['num_classes'] = num_classes
