import cv2
import logging
import numpy as np


class Metric:
    def __init__(self, encoding, history=1):
        assert history > 0, 'At least one step backward in history consideration'
        self.encoding = encoding
        self.history = history

    def __call__(self, tracklets, detection_features):
        """
        Calculate the similarity matrix between tracklets and detections.
        :param tracklets: A list of active tracklets, assuming its size is (m).
        :param detection_features: A list of N feature dicts (encoded by detections).
        :return: A matrix of shape (m, n) and a list of detection features.
        """
        matrix = np.zeros([len(tracklets), len(detection_features)])
        for i in range(len(tracklets)):
            for j in range(len(detection_features)):
                affinities = []
                for k in range(min(self.history, len(tracklets[i].feature_history))):
                    affinities.append(self.similarity(tracklets[i].feature_history[-k - 1][1], detection_features[j]))
                matrix[i][j] = sum(affinities) / len(affinities)
        self._log_affinity_matrix(matrix, tracklets, self.encoding)
        return matrix

    def similarity(self, tracklet_feature, detection_feature):
        raise NotImplementedError('Extend the Metric class to implement your own distance metric.')

    def _log_affinity_matrix(self, matrix, tracklets, metric_name):
        logger = logging.getLogger('MOT')
        logger.debug('Metric {}:'.format(metric_name))
        logger.debug('==============================')
        table_head = '| T\\D\t|'
        if len(matrix) > 0:
            for i in range(len(matrix[0])):
                table_head += ' {:5d}\t|'.format(i)
        logger.debug(table_head)
        for index, line in enumerate(matrix):
            text = '| {} \t| '.format(tracklets[index].id)
            for i in line:
                text += '{:.3f}\t| '.format(i)
            logger.debug(text)
        logger.debug('==============================')

        if logger.level <= logging.DEBUG and hasattr(logger, 'display'):
            self._display_affinity_matrix(matrix, metric_name, [tracklet.id for tracklet in tracklets])

    def _display_affinity_matrix(self, matrix, metric_name, tracklet_ids):
        if matrix is not None and matrix.shape[0] > 0:
            matrix = cv2.copyMakeBorder(matrix, 1, 0, 1, 0, borderType=cv2.BORDER_CONSTANT, value=0)
            img = cv2.resize(matrix, (256, 256), interpolation=cv2.INTER_NEAREST)
            step = 256 / (len(tracklet_ids) + 1)
            for index, tracklet_id in enumerate(tracklet_ids):
                img = cv2.putText(img, '{}'.format(tracklet_id), (2, int((index + 1.6) * step)),
                                  cv2.FONT_HERSHEY_SIMPLEX,
                                  0.4, 1, thickness=1)

            cv2.imshow('Affinity matrix of metric {}'.format(metric_name), img)
            cv2.waitKey(1)
