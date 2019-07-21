class Metric:
    def __init__(self):
        raise NotImplementedError('Extend the Metric class to implement your own distance metric.')

    def __call__(self, tracklets, detected_boxes, img):
        """
        Calculate the similarity matrix between tracklets and detections.
        :param tracklets: A list of active tracklets, assuming its size is (m).
        :param detected_boxes: A numpy array of detected boxes with shape (n, 5).
        :param img: A numpy array of shape (h, w, 3). The new frame.
        :return: A matrix of shape (m, n) and a list of detection features.
        """
        raise NotImplementedError('Extend the Metric class to implement your own distance metric.')
