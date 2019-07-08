class Metric:
    def __init__(self):
        raise NotImplementedError('Extend the Metric class to implement your own distance metric.')

    def __call__(self, a, b):
        raise NotImplementedError('Extend the Metric class to implement your own distance metric.')
