from .metric import Metric


class MultipleMetrics(Metric):
    def __init__(self, metric1, metric2):
        super(MultipleMetrics).__init__()
        self.metric1 = metric1
        self.metric2 = metric2

    def __call__(self, tracklets, detected_boxes, img):
        similarity_matrix1, features1 = self.metric1(tracklets, detected_boxes, img)
        similarity_matrix2, features2 = self.metric2(tracklets, detected_boxes, img)
        similarity_matrix = similarity_matrix1.copy()
        for i in range(len(similarity_matrix1)):
            for j in range(len(similarity_matrix1[0])):
                similarity_matrix[i][j] = self.combine(similarity_matrix1[i][j], similarity_matrix2[i][j])
        return similarity_matrix, [(features1[i], features2[i]) for i in range(len(detected_boxes))]

    def combine(self, similarity1, similarity2):
        raise NotImplementedError(
            'Extend the MultipleMetrics class to implement your own multi-cue-combination method.')


class ProductMetrics(MultipleMetrics):
    def combine(self, similarity1, similarity2):
        return similarity1 * similarity2


class SummationMetrics(MultipleMetrics):
    def combine(self, similarity1, similarity2):
        return similarity1 + similarity2
