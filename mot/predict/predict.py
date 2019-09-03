class Predictor:
    def __init__(self):
        raise NotImplementedError('Extend the Predictor class to implement your own prediction method.')

    def __call__(self, tracklet):
        return self.predict(tracklet)

    def initiate(self, tracklet):
        """
        Initiate a tracklet with its initial mean and covariance.
        """
        raise NotImplementedError('Extend the Predictor class to implement your own prediction method.')

    def update(self, tracklet):
        """
        Update a tracklet with its last box coordinates.
        """
        raise NotImplementedError('Extend the Predictor class to implement your own prediction method.')

    def predict(self, tracklet):
        """
        Predict state in the following time step.
        :return: Predicted state in the following time step.
        """
        raise NotImplementedError('Extend the Predictor class to implement your own prediction method.')
