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
        all_tracklet_features = []
        for (camID, localID), tracklet in self.tracklets.items():
            all_tracklet_features.append(tracklet.sample_features())
        all_tracklet_features = np.vstack(all_tracklet_features)
        if len(all_tracklet_features) > 1:
            clusterIDs = hierarchical_cluster(all_tracklet_features, self.max_cluster_distance, criterion='distance')

            logging.getLogger('MTMCT').info(
                'Shrinking Identity #{}\'s {} features into {}'.format(self.globalID,
                                                                       len(all_tracklet_features),
                                                                       len(np.unique(clusterIDs))))

            features = []
            for clusterID in np.unique(clusterIDs):
                inds = np.where(clusterIDs == clusterID)[0]
                feature = np.average(all_tracklet_features[inds], axis=0)
                features.append(feature)
            self.features = features
        else:
            self.features = [feature for feature in all_tracklet_features]
        return self.features
