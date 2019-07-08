class Encoder:
    def __init__(self):
        raise NotImplementedError('Extend the Encoder class to implement your own feature extractor.')

    def __call__(self, boxes, img):
        raise NotImplementedError('Extend the Encoder class to implement your own feature extractor.')
