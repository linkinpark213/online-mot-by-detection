class Matcher:
    def __init__(self):
        raise NotImplementedError('Extend the Matcher class to implement your own association algorithm.')

    def __call__(self, tracks, detections):
        raise NotImplementedError('Extend the Matcher class to implement your own association algorithm.')
