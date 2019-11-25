class Detector:
    def __init__(self):
        raise NotImplementedError('Extend the Detector class to implement your own detector.')

    def __call__(self, img):
        """
        Detect all objects in an image.
        :param img: A numpy array of shape (H, W, 3)
        :return: A list of N Detection objects.
        """
        raise NotImplementedError('Extend the Detector class to implement your own detector.')


class Detection:
    def __init__(self, box, score, mask=None):
        self.box = box
        self.score = score
        self.mask = mask
