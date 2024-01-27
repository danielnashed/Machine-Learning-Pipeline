
#from null_model import null_model

# input is inference data
# output is predictions
class Model:
    def __init__(self, config=None):
        self.config = config # setup model based on config file proided 
        self.model = [] # model

    def select(self):
        # Select the model
        #self.model = null_model.NullModel() # model
        self.model = Model(self)
        print('Selecting the model...')
        return self.model

    def predict(self, data):
        # Predict
        predictions = []
        print('Predicting...')
        return predictions