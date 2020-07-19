import time
import logging
import numpy as np
from typing import List

from mot.utils.hcc import hierarchical_cluster


class Tracklet:
    def __init__(self, camID: int, localID: int, feature: np.ndarray, n_feature_samples: int):
        self.camID: int = camID
        self.localID: int = localID
        self.globalID: int = -1
        self.features: List[np.ndarray] = [feature]
        self.n_feature_samples = n_feature_samples
        self.created_time = time.time()
        self.last_active_time = time.time()
        self.last_sample_time = 0

    def sample_features(self):
        if self.last_sample_time == self.last_active_time:
            # Not updated since last sampling
            return np.array(self.features)

        features = np.array(self.features)
        if len(features) > self.n_feature_samples:
            clusterIDs = hierarchical_cluster(np.array(features), self.n_feature_samples,
                                              linkage_method='average', criterion='maxclust')
            tracklet_features = []
            for clusterID in np.unique(clusterIDs):
                inds = np.where(clusterIDs == clusterID)[0]
                feature = np.average(features[inds], axis=0)
                tracklet_features.append(feature)

            logging.getLogger('MTMCT').debug('Shrinking target c{}-t{}\'s {} features into {}'.format(self.camID,
                                                                                                      self.localID,
                                                                                                      len(
                                                                                                          self.features),
                                                                                                      len(
                                                                                                          tracklet_features)))
            self.features = tracklet_features
            self.last_sample_time = self.last_active_time
        return np.array(self.features)

    def add_feature(self, feature):
        self.last_active_time = time.time()
        self.features.append(feature)
