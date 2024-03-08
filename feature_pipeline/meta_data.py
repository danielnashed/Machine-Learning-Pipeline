
"""
This module contains the MetaData class which is used to store metadata of the dataset such as 
column names, positive class, and number of classes in addition to the data itself. It is used 
to pass the metadata to the model and learner classes.

The MetaData class contains the following attributes:
    - data: the data in the dataset
    - meta: a dictionary to store the metadata

The MetaData class contains the following methods:
    - set_meta: set the metadata of the dataset
"""
class MetaData():
    def __init__(self):
        self.data = None
        self.meta = {}

    def set_meta(self, data, column_names=None, positive_class=None, num_classes=None):
        self.data = data
        self.meta['column_names'] = column_names # column names for the dataset
        self.meta['pos_class'] = positive_class # positive class for binary classification
        self.meta['num_classes'] = num_classes # number of classes for multi-class classification
