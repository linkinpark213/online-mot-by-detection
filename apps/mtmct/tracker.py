import cv2
import logging
import numpy as np
from typing import List, Union, Tuple, Dict

from mot.utils.config import Config
from mot.structures import Tracklet
from mot.utils.io import ImagesCapture
from mot.utils.hcc import hierarchical_cluster
from mot.tracker import build_tracker, Tracker, TrackerState


def _sample_tracklet_features(tracklet: Tracklet, n_feature_samples: int):
    recent_detections = tracklet.detection_history[-len(tracklet.feature_history):]
    p = np.array([detection.score for _, detection in recent_detections])

    all_recent_features = np.stack([feature[1]['openreid'] for feature in tracklet.feature_history])
    M = 1 - np.matmul(all_recent_features, all_recent_features.T)
    M = np.clip(M, 0, 1)

    clusterIDs = hierarchical_cluster(M, n_feature_samples, linkage_method='average', criterion='maxclust')
    tracklet_features = []
    for clusterID in np.unique(clusterIDs):
        inds = np.where(clusterIDs == clusterID)[0]
        feature = np.sum((all_recent_features[inds].T * (p[inds] / np.sum(p[inds]))).T, axis=0, keepdims=True)
        tracklet_features.append(feature)

    tracklet_features = np.vstack(tracklet_features)
    print('Shrinking target #{}\'s {} features into {}'.format(tracklet.id, len(M), len(tracklet_features)))
    return tracklet_features


class Identity:
    def __init__(self, globalID: int, camID: int, localID: int, tracklet: Tracklet, n_feature_samples: int = 8,
                 max_cluster_distance: float = 0.1):
        self.globalID: int = globalID
        self.end_time: int = tracklet.detection_history[-1][0]
        self.tracklets: Dict[Tuple[int, int], Tracklet] = {(camID, localID): tracklet}
        self.max_cluster_distance = max_cluster_distance
        self.n_feature_samples: int = n_feature_samples

        recent_detections = tracklet.detection_history[-len(tracklet.feature_history):]
        detecton_scores = np.array([detection.score for _, detection in recent_detections])
        best_ind = int(np.argmax(detecton_scores))
        self.images: Dict[int, np.ndarray] = {camID: tracklet.feature_history[best_ind][1]['patch'].copy()}

        if tracklet.sample_features is None:
            tracklet.update_sample_features()
        self.features = tracklet.sample_features

        del tracklet.feature_history
        del tracklet.feature

    def merge(self, others: List):
        for other in others:
            for k, v in other.tracklets.items():
                self.tracklets[k] = v
                self.end_time = max(self.end_time, v.detection_history[-1][0])
        return self

    def add_tracklet(self, tracklet: Tracklet, camID: int):
        tracklet.globalID = self.globalID
        self.tracklets[(camID, tracklet.id)] = tracklet
        self.end_time = max(self.end_time, tracklet.last_active_frame_num)
        if tracklet.id not in self.images:
            recent_detections = tracklet.detection_history[-len(tracklet.feature_history):]
            detecton_scores = np.array([detection.score for _, detection in recent_detections])
            best_ind = int(np.argmax(detecton_scores))
            self.images[camID] = tracklet.feature_history[best_ind][1]['patch'].copy()

    def is_active(self):
        for _, tracklet in self.tracklets.items():
            if not tracklet.is_finished():
                return True
        return False


class MCTracker:
    # Infinite distance between local tracklets.
    INF = 1

    def __init__(self, stracker_config: Config, captures: Dict[int, Union[ImagesCapture, cv2.VideoCapture]],
                 cluster_freq: int = 60, max_ttl: int = 6000, max_local_overlap: int = 10,
                 max_reid_distance: float = 0.25, max_cluster_distance: float = 0.1, n_feature_samples: int = 8):
        # Single camera tracker for all cameras.
        self.stracker: Tracker = build_tracker(stracker_config)
        # Tracker states for switching.
        self.stracker_states: Dict[int, TrackerState] = {camID: TrackerState() for camID, _ in captures.items()}
        # Corresponding captures (ImagesCapture or cv2.VideoCapture objects).
        self.captures: Dict[int, Union[ImagesCapture, cv2.VideoCapture]] = captures
        # Global tracklet pool. globalID => Identity.
        self.identity_pool: Dict[int, Identity] = {}
        # Global ID for allocation
        self.max_id: int = 0
        # Global frame number.
        self.frame_num: int = 0
        # Frequency to perform global clustering.
        self.cluster_freq: int = cluster_freq
        # Maximum time-to-live of any tracklet.
        self.max_ttl: int = max_ttl
        # Maximum local overlap time.
        self.max_local_overlap = max_local_overlap
        # Re-identification distance threshold.
        self.max_reid_distance = max_reid_distance
        # Local clustering distance threshold.
        self.max_cluster_distance = max_cluster_distance
        # Number of feature samples for each new identity
        self.n_feature_samples = n_feature_samples
        # Feature gallery (instead of image gallery)
        self.feature_gallery = np.ndarray((0, 256), dtype=np.float)
        # Corresponding IDs for each line of feature gallery
        self.id_gallery = np.ndarray((0,), dtype=np.int)
        # Logger
        self.logger: logging.Logger = logging.getLogger('MTMCT')

    def tick(self):
        self.frame_num += 1
        for camID, cap in self.captures.items():

            ret, frame = cap.read()
            if not ret:
                return False
            self.stracker.state = self.stracker_states[camID]
            self.stracker.tick(frame)

        self.update_identity_pool()
        if self.frame_num % self.cluster_freq == 0:
            # self.update_tracklet_features()
            self.query_identities()
        return True

    def update_identity_pool(self):
        """
        Update identities with new tracklets added, or initiate new identities with single-cam tracklets.
        """
        for camID, stracker_state in self.stracker_states.items():
            while len(stracker_state.tracklets_finished) > 0:
                tracklet = stracker_state.tracklets_finished.pop()
                if tracklet.globalID >= 0:
                    # Tracklet has been re-identified
                    identity = self.identity_pool[tracklet.globalID]

                    # Update Identity features using this tracklet with randomly chosen features
                    # recent_detections = tracklet.detection_history[-len(tracklet.feature_history):]
                    # p = np.array([detection.score for _, detection in recent_detections])
                    # p = p / np.sum(p)
                    # inds = np.random.choice(range(len(tracklet.feature_history)), self.n_feature_samples, replace=False,
                    #                         p=p)
                    # new_features = np.stack([tracklet.feature_history[ind][1]['openreid'].copy() for ind in inds])

                    # New update method: Shrink the feature history size by HCC
                    if tracklet.sample_features is None:
                        tracklet.update_sample_features()
                    identity.features = np.vstack((identity.features, tracklet.sample_features))

                else:
                    # Tracklet hasn't been re-identified yet, start a new identity
                    if tracklet.time_lived >= self.n_feature_samples:
                        self.max_id += 1
                        tracklet.globalID = self.max_id
                        self.identity_pool[self.max_id] = Identity(self.max_id, camID, tracklet.id, tracklet,
                                                                   n_feature_samples=self.n_feature_samples)
                        self.logger.info('Initiating global ID #{} from tracklet (cam = {}, localID = {})'.format(
                            self.max_id, camID, tracklet.id
                        ))

    # def update_gallery_features(self):
    #     """
    #     Update feature gallery and corresponding features for queries.
    #     """
    #     if len(self.identity_pool) == 0:
    #         return
    #     features, globalIDs = [], []
    #     for globalID, identity in self.identity_pool.items():
    #         features.append(identity.features)
    #         globalIDs.extend([globalID] * len(identity.features))
    #     self.feature_gallery = np.vstack(features)
    #     self.id_gallery = np.array(globalIDs)
    #     assert len(self.feature_gallery) == len(self.id_gallery)

    def clear_inactive_identities(self):
        """
        Clear old identities.
        """
        globalIDs = list(self.identity_pool.keys())
        end_times = np.array([(self.frame_num if identity.is_active() else identity.end_time) for _, identity in
                              self.identity_pool.items()])
        inds_to_clear = np.where(self.frame_num - end_times > self.max_ttl)[0]
        inds_to_clear = np.sort(inds_to_clear)[::-1]
        identities = []
        for ind in inds_to_clear:
            globalID = globalIDs[int(ind)]
            identities.append((globalID, self.identity_pool.pop(globalID)))
        return identities

    def terminate(self):
        for camID, state in self.stracker_states.items():
            self.stracker.state = state
            self.stracker.terminate()
        self.update_identity_pool()
        # self.update_gallery_features()
        self.query_identities(query_all=True)

    def query_identities(self, query_all: bool = False):
        if len(self.identity_pool) == 0:
            return
        camIDs = []
        tracklets = []
        M = []
        # Build up query features
        for i, (camID, state) in enumerate(self.stracker_states.items()):
            for tracklet in state.tracklets_active:
                # Only consider tracklets that are long enough
                if tracklet.globalID == -1 and (query_all or len(tracklet.feature_history) >= self.n_feature_samples):
                    camIDs.append(camID)
                    tracklets.append(tracklet)

                    # Sample features from feature history
                    if tracklet.sample_features is None:
                        tracklet.update_sample_features()
                    tracklet_features = tracklet.sample_features

                    dists = []
                    for j, (globalID, identity) in enumerate(self.identity_pool.items()):
                        dists.append(np.average(1 - np.matmul(tracklet_features, identity.features.T).reshape(-1)))
                        for key, other in identity.tracklets.items():
                            # No long overlap allowed
                            if tracklet.time_overlap(other) > self.max_local_overlap:
                                dists[-1] = 1
                                break
                    M.append(dists)

        # Perform greedy matching
        M = np.array(M)
        flattened = np.reshape(M, (-1))
        inds = list(np.argsort(flattened))
        row_ind, col_ind = [], []
        for ind in inds:
            if flattened[ind] > self.max_reid_distance:
                break
            row, col = ind // len(self.identity_pool), ind % len(self.identity_pool)
            if row not in row_ind and col not in col_ind:
                row_ind.append(row)
                col_ind.append(col)

        # Merge tracklets into identities
        globalIDs = list(self.identity_pool.keys())
        for row, col in zip(row_ind, col_ind):
            identity = self.identity_pool[globalIDs[col]]
            self.logger.info(
                'Local tracklet (cam = {}, localID = {}) matches global ID {} (distance = {})'.format(camIDs[row],
                                                                                                      tracklets[
                                                                                                          row].id,
                                                                                                      identity.globalID,
                                                                                                      M[row][col]))
            identity.add_tracklet(tracklets[row], camIDs[row])

    def _time_overlap(self, period1: Tuple[int, int], period2: Tuple[int, int]):
        overlap = min(period2[1], period1[1]) - max(period2[0], period1[0])
        return overlap if overlap > 0 else 0

    def _cosine_distance(self, vector1, vector2):
        return 1 - np.dot(vector1, vector2) / ((np.linalg.norm(vector1) * np.linalg.norm(vector2)) + 1e-16)

    def _identity_distance(self, identity1: Identity, identity2: Identity):
        for (cam1, localID1), tracklet1 in identity1.tracklets.items():
            for (cam2, localID2), tracklet2 in identity2.tracklets.items():
                if cam1 == cam2 and self._time_overlap(
                        (tracklet1.detection_history[0][0], tracklet1.detection_history[-1][0]),
                        (tracklet2.detection_history[0][0], tracklet2.detection_history[-1][0])
                ) > self.max_local_overlap:
                    return self.INF

        features1 = [tracklet.feature['openreid'] for (_, _), tracklet in identity1.tracklets.items()]
        features2 = [tracklet.feature['openreid'] for (_, _), tracklet in identity2.tracklets.items()]
        return max(
            [max([self._cosine_distance(feature1, feature2) for feature2 in features2]) for feature1 in features1])
