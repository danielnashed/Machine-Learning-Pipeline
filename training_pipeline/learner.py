
# input is processed data and model
# output is model
class Learner:
    def __init__(self, model):
        self.model = model # takes model which has already been setup using config file in model.py
        #self.config = config + '.config' # setup file for model
        #self.model = config + '.model' # model file

    def train_model(self, data):
        # Train the model
        print('Training the model...')
        return None

    def validate_model(self, data):
        # Validate the model
        print('Validating the model...')
        return None

    def export_model(self):
        # Export the model
        print('Exporting the model...')
        return None

    def export_logs(self):
        # Export the logs
        print('Exporting the logs...')
        return None

    def learn(self, data):
        # Run the learner
        self.train_model(data)
        self.validate_model(data)
        self.export_model()
        self.export_logs()
        return self.model