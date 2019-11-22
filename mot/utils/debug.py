import cv2
import logging


def log_affinity_matrix(matrix, tracklets, metric_name):
    logger = logging.getLogger('MOT')
    logger.debug('Metric {}:'.format(metric_name))
    logger.debug('==============================')
    logger.debug('| T\\D\t|')
    for index, line in enumerate(matrix):
        text = '| {} \t| '.format(tracklets[index].id)
        for i in line:
            text += '{:.3f}\t| '.format(i)
        logger.debug(text)
    logger.debug('==============================')

    if logger.level <= logging.DEBUG and hasattr(logger, 'display'):
        display_affinity_matrix(matrix, metric_name, [tracklet.id for tracklet in tracklets])


def display_affinity_matrix(matrix, metric_name, tracklet_ids):
    if matrix is not None and matrix.shape[0] > 0:
        matrix = cv2.copyMakeBorder(matrix, 1, 0, 1, 0, borderType=cv2.BORDER_CONSTANT, value=0)
        img = cv2.resize(matrix, (256, 256), interpolation=cv2.INTER_NEAREST)
        step = 256 / (len(tracklet_ids) + 1)
        for index, tracklet_id in enumerate(tracklet_ids):
            img = cv2.putText(img, '{}'.format(tracklet_id), (2, int((index + 1.6) * step)), cv2.FONT_HERSHEY_SIMPLEX,
                              0.4, 1, thickness=1)

        cv2.imshow('Affinity matrix of metric {}'.format(metric_name), img)
        cv2.waitKey(1)
