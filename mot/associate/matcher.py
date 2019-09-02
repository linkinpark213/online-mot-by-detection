class Matcher:
    def __init__(self, metric):
        self.metric = metric

    def __call__(self, tracklets, detection_boxes, img):
        raise NotImplementedError('Extend the Matcher class to implement your own association algorithm.')
