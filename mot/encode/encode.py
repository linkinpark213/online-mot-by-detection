class Encoder:
    def __init__(self):
        self.name = 'encoding'

    def __call__(self, detections, img):
        raise NotImplementedError('Extend the Encoder class to implement your own feature extractor.')
