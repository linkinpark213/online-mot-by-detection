import cv2
import logging
import numpy as np
from typing import List, Dict, Union
from abc import ABCMeta, abstractmethod

from mot.utils import Registry
from mot.structures import Tracklet

__all__ = ['Metric', 'METRIC_REGISTRY', 'build_metric']

METRIC_REGISTRY = Registry('metrics')


class Metric(metaclass=ABCMeta):
    def __init__(self, cfg) -> None:
        if cfg is not None:
            self.history = 1
            if hasattr(cfg, 'history'):
                assert cfg.history > 0, 'At least one step backward (last frame) in history consideration'
                self.history = cfg.history
            self.encoding = cfg.encoding
            self.name = cfg.name if hasattr(cfg, 'name') else cfg.encoding

    def __call__(self, tracklets: List[Tracklet], detection_features: List[Dict]) -> Union[np.ndarray, List[List[int]]]:
        return self.affinity_matrix(tracklets, detection_features)

    def affinity_matrix(self, tracklets: List[Tracklet], detection_features: List[Dict]) -> Union[
        np.ndarray, List[List[float]]]:
        """
        Calculate the similarity matrix between tracklets and detections.

        Args:
            tracklets: A list of active tracklets, assuming its size is (m).
            detection_features: A list of N feature dicts (encoded by detections).

        Returns:
            A np array of shape (m, n) and a list of detection features.
        """
        matrix = np.zeros([len(tracklets), len(detection_features)])
        for i in range(len(tracklets)):
            for j in range(len(detection_features)):
                affinities = []
                for k in range(min(self.history, len(tracklets[i].feature_history))):
                    affinities.append(self.similarity(tracklets[i].feature_history[-k - 1][1], detection_features[j]))
                matrix[i][j] = sum(affinities) / len(affinities)
        self._log_affinity_matrix(matrix, tracklets, self.name)
        return matrix

    @abstractmethod
    def similarity(self, tracklet_feature: Dict, detection_feature: Dict) -> float:
        pass

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


def build_metric(cfg):
    return METRIC_REGISTRY.get(cfg.type)(cfg)
