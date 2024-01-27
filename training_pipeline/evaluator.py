# input is predictions and labels
# output is evaluation metrics
class Evaluator:
    def __init__(self, config):
        self.config = config # config file for model contains what metrics to use for evaluation

    def evaluate(self, labels, predictions):
        # Evaluate the model
        metrics = 0
        print('Evaluating the model...')
        return metrics