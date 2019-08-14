class Detector:
    def __init__(self):
        raise NotImplementedError('Extend the Detector class to implement your own detector.')

    def __call__(self, img):
        """
        Detect all objects in an image.
        :param img: A numpy array of shape (H, W, 3)
        :return: A numpy array of shape (N, 5) or an empty list. The (x1, y1, x2, y2, conf) of all detected objects.
        """
        raise NotImplementedError('Extend the Detector class to implement your own detector.')
