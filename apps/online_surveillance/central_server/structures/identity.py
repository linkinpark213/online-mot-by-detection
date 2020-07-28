import time
import logging
import numpy as np
from typing import List, Tuple, Dict

from mot.utils.hcc import hierarchical_cluster

from .tracklet import Tracklet


class Identity:
    def __init__(self, globalID: int, tracklet: Tracklet, max_cluster_distance: float = 0.1):
        self.globalID: int = globalID
        self.created_time: float = tracklet.created_time
        self.last_active_time = tracklet.last_active_time
        self.tracklets: Dict[Tuple[int, int], Tracklet] = {(tracklet.camID, tracklet.localID): tracklet}
        self.max_cluster_distance = max_cluster_distance
        self.features: List[np.ndarray] = []
        self.last_active_time = time.time()
        self.last_sample_time = 0

    def is_overlapping(self, tracklet: Tracklet):
        for (camID, localID), _tracklet in self.tracklets.items():
            if camID == tracklet.camID and max(_tracklet.created_time, tracklet.created_time) < min(
                    _tracklet.last_active_time, tracklet.last_active_time):
                return True
        return False

    def add_tracklet(self, tracklet: Tracklet):
        tracklet.globalID = self.globalID
        self.tracklets[(tracklet.camID, tracklet.localID)] = tracklet
        self.created_time = min(self.created_time, tracklet.created_time)
        self.last_active_time = max(self.last_active_time, tracklet.last_active_time)

    def sample_features(self):
        # If no tracklet updated after last sampling, do not waste time sampling.
        if max([tracklet.last_active_time for _, tracklet in self.tracklets.items()]) <= self.last_sample_time:
            return self.features

        # Stack all tracklet feature samples from each tracklet.
        all_tracklet_features = []
        for _, tracklet in self.tracklets.items():
            all_tracklet_features.append(tracklet.sample_features())
        all_tracklet_features = np.vstack(all_tracklet_features)

        if len(all_tracklet_features) > 1:
            clusterIDs = hierarchical_cluster(all_tracklet_features, self.max_cluster_distance, criterion='distance')

            logging.getLogger('MTMCT').info(
                'Clustering identity #{}\'s {} features into {}'.format(self.globalID,
                                                                        len(all_tracklet_features),
                                                                        len(np.unique(clusterIDs))))

            features = []
            for clusterID in np.unique(clusterIDs):
                inds = np.where(clusterIDs == clusterID)[0]
                # Average of all feature vectors?
                # feature = np.average(all_tracklet_features[inds], axis=0)
                # A random feature vector?
                feature = all_tracklet_features[np.random.choice(inds)]
                features.append(feature)
            self.features = features
        else:
            self.features = [feature for feature in all_tracklet_features]

        self.last_sample_time = time.time()

        return self.features
