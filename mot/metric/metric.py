class Metric:
    def __init__(self):
        raise NotImplementedError('Extend the Metric class to implement your own distance metric.')

    def __call__(self, a, b):
        """
        Calculate the similarity between a tracklet and a detection.
        :param a: Feature of the tracklet.
        :param b: Feature of the detection.
        :return: A floating point number indicating the similarity of a tracklet and a detection.
        """
        raise NotImplementedError('Extend the Metric class to implement your own distance metric.')
