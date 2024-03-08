import operator
import copy
from math import isnan
import configparser
from inference_pipeline.decision_tree import node as TreeNode
from training_pipeline import evaluator as __evaluator__
from inference_pipeline.decision_tree import tree_visualizer as TreeVisualizer


class DecisionTreePruner():
    def __init__(self, meta_tree):
        self.tree = meta_tree.function # get the decision tree
        self.validation_set = meta_tree.validation_set # get the validation set
        self.prediction_type = meta_tree.prediction_type # get the prediction type
        self.non_grandparents = meta_tree.non_grandparents # initialize list of non-grandparent nodes
        self.num_nodes_before_pruning = meta_tree.id # initialize number of nodes before pruning
        self.num_nodes_after_pruning = 1 # initialize number of nodes after pruning
        # initialize max depth before pruning and after pruning
        self.max_depth_before_pruning = self.find_max_depth(self.tree)
        self.max_depth_after_pruning = None
        self.performance_metric_before_pruning = None # initialize performance metric before pruning
        self.metric_name = None # initialize metric name
        self.metric = None # initialize metric 
        # initialize dictionary to map operator based on split criterion condition
        self.ops =  {'>': operator.gt, '<': operator.lt}
        self.op = None
        self.predict = meta_tree.predict # get the predict function
        self.is_grandparent = meta_tree.is_grandparent # get the is_grandparent function
        self.evaluator = __evaluator__.Evaluator(meta_tree, self.metrics_for_pruning()) # initialize the evaluator class

        # # for debugging only 
        # self.column_names = meta_tree.column_names
        # self.visualizer = TreeVisualizer.DecisionTreeVisualizer(self.column_names, self.prediction_type, 'before_pruning.gv')
        # self.visualizer.draw_tree(self.tree) # before pruning

    def set_eval_metric(self):
        # set the metric name and operator based on prediction type
        if self.prediction_type == 'classification':
            self.metric_name = 'accuracy'
            self.op = '>'
            config_string = """
            [evaluation_metrics]
            accuracy: 1
            precision: 0
            recall: 0
            f1: 0
            r2: 0
            mse: 0
            mae: 0
            rmse: 0
            """
        elif self.prediction_type == 'regression':
            self.metric_name = 'mse'
            self.op = '<'
            config_string = """
            [evaluation_metrics]
            accuracy: 0
            precision: 0
            recall: 0
            f1: 0
            r2: 0
            mse: 1
            mae: 0
            rmse: 0
            """
        return config_string

    def metrics_for_pruning(self):
        # create a new ConfigParser object
        config = configparser.ConfigParser()
        # read configuration data from emulated string
        config.read_string(self.set_eval_metric())
        return config
    
    def find_max_depth(self, root):
        # if the node has no children, return the depth
        if not root.children:
            return root.depth
        # if the node has children, recursively find the max depth
        if root.children:
            depths = []
            for child in root.children:
                depths.append(self.find_max_depth(child))
            return max(depths)
        return None

    def search_node(self, node_id, root):
        # if node id is found, return the node
        if root.id == node_id:
            return root
        # if node id is not found, recursively search for the node in the children
        if root.children:
            for child in root.children:
                node = self.search_node(node_id, child)
                if node:
                    return node
        return None
    
    def post_order_traversal(self, root):
        if root.children:
            for child in root.children:
                self.post_order_traversal(child)
            root.grandparent = self.is_grandparent(root, root.children)
            # if the node is not a grandparent and has not been visited, add it to the list of non-grandparents
            if not root.grandparent and root.visited == False:
                self.non_grandparents.append(root.id)
        return None
    
    def update_non_grandparents(self, root):
        current_non_grandparents = self.non_grandparents
        self.non_grandparents = [] # reset non-grandparents
        self.post_order_traversal(root)
        new_non_grandparents = self.non_grandparents
        # if a change in non-grandparents is detected and list is not empty, then keep pruning
        if (current_non_grandparents != new_non_grandparents) and (len(new_non_grandparents) > 0):
            return True
        return False
    
    def count_nodes(self, root):
        if root.children:
            for child in root.children:
                self.num_nodes_after_pruning += 1
                self.count_nodes(child)
        return None
    
    def print_stats(self):
        print(f'        Number of nodes --> before pruning: {self.num_nodes_before_pruning}, after pruning: {self.num_nodes_after_pruning}')
        print(f'        Max depth --> before pruning: {self.max_depth_before_pruning}, after pruning: {self.max_depth_after_pruning}')
        print(f'        {self.metric_name} --> before pruning: {self.performance_metric_before_pruning}, after pruning: {self.metric}')
        return None
    
    def prune_tree(self):
        # evaluate original tree on validation set
        x_val, y_val = self.validation_set
        y_pred = self.predict(x_val)
        self.metric = self.evaluator.evaluate(y_val, y_pred)[self.metric_name]
        self.performance_metric_before_pruning = self.metric
        keep_pruning = True # flag to keep pruning
        counter = 0 # counter to prevent infinite loop

        # keep pruning until no more non-grandparent nodes are left or we cannot improve performance
        while keep_pruning and counter < 10:
            counter += 1
            print(f'        Pruning decision tree, iteration: {counter}')
            # iterate over all non-grandparent nodes
            for non_grandparent in self.non_grandparents:
                # create a new copy of the decision tree
                new_tree = copy.deepcopy(self.tree)
                node = self.search_node(non_grandparent, new_tree)
                # prune the node
                node.children = None
                node.feature = None
                node.threshold = None
                node.grandparent = None
                node.visited = True
                # test new tree on validation set
                y_pred = self.predict(x_val, new_tree)
                # evaluate performance of new tree
                metric = self.evaluator.evaluate(y_val, y_pred)[self.metric_name]
                # if classification, select higher accuracy. If regression, select lower mean squared error
                if (not isnan(metric)) and (self.ops[self.op](metric, self.metric)):
                    self.tree = new_tree
                    self.metric = metric
                    # otherwise, mark the node as visited in original tree
                else:
                    node = self.search_node(non_grandparent, self.tree)
                    node.visited = True
             # after pruning, update the list of non-grandparent nodes
            keep_pruning = self.update_non_grandparents(self.tree)
        # calculate new max depth after pruning
        self.max_depth_after_pruning = self.find_max_depth(self.tree)
        # recalculate new number of nodes after pruning
        self.count_nodes(self.tree)
        self.print_stats() # print stats of pruning 

        # # for debugging only
        # self.visualizer = TreeVisualizer.DecisionTreeVisualizer(self.column_names, self.prediction_type, 'after_pruning.gv')
        # self.visualizer.draw_tree(self.tree) # before pruning


        return self.tree