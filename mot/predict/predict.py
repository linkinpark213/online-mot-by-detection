class Predictor:
    def __init__(self):
        raise NotImplementedError('Extend the Predictor class to implement your own prediction method.')

    def __call__(self, tracklets, img):
        return self.predict(tracklets, img)

    def initiate(self, tracklets):
        """
        Initiate tracklets' states that are used by the predictor.
        """
        raise NotImplementedError('Extend the Predictor class to implement your own prediction method.')

    def update(self, tracklets):
        """
        Update tracklets' states that are used by the predictor.
        """
        raise NotImplementedError('Extend the Predictor class to implement your own prediction method.')

    def predict(self, tracklets, img):
        """
        Predict state in the following time step.
        :return: A list of Prediction objects corresponding to the tracklets.
        """
        raise NotImplementedError('Extend the Predictor class to implement your own prediction method.')


class Prediction:
    def __init__(self, box, score):
        self.box = box
        self.score = score
