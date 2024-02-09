# input is evaluation metrics
# output is dashboard visuals
class Dashboard:
    def __init__(self, config):
        self.config = config

    def visualize(self, metrics, logs):
        # Visualize the data
        print('Visualizing the data...')
        self.export_dashboard(metrics, logs)
        return None

    def export_dashboard(self, metrics, logs):
        # Export the dashboard
        print('Exporting the dashboard...')
        return None