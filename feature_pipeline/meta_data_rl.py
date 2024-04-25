
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

    def set_meta(self, data, start_state=None, goal_state=None, forbidden_state=None):
        self.data = data
        self.meta['start_state'] = start_state # start state for agent
        self.meta['goal_state'] = goal_state # goal state for agent
        self.meta['forbidden_state'] = forbidden_state # forbidden state for agent
