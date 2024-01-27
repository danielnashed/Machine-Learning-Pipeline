# input is evaluation metrics
# output is dashboard visuals
class Dashboard:
    def __init__(self, config):
        self.config = config

    def visualize(self, evaluation):
        # Visualize the data
        print('Visualizing the data...')
        self.export_dashboard(evaluation)
        return None

    def export_dashboard(self, data):
        # Export the dashboard
        print('Exporting the dashboard...')
        return None