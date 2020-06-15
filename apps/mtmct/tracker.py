import cv2
import numpy as np
from typing import List, Union, Tuple, Dict

from mot.utils.config import Config
from mot.structures import Tracklet
from mot.utils.io import ImagesCapture
from mot.tracker import build_tracker, Tracker, TrackerState


class Identity:
    def __init__(self, globalID: int, camID: int, localID: int, tracklet: Tracklet, n_feature_samples: int = 8):
        self.globalID: int = globalID
        self.end_time: int = tracklet.detection_history[-1][0]
        self.tracklets: Dict[Tuple[int, int], Tracklet] = {(camID, localID): tracklet}

        recent_detections = tracklet.detection_history[-len(tracklet.feature_history):]
        p = np.array([detection.score for _, detection in recent_detections])
        p = p / np.sum(p)
        inds = np.random.choice(range(len(tracklet.feature_history)), n_feature_samples, replace=False, p=p)
        self.features = np.stack([tracklet.feature_history[ind][1]['openreid'].copy() for ind in inds])

        del tracklet.feature_history
        del tracklet.feature

    def merge(self, others: List):
        for other in others:
            for k, v in other.tracklets.items():
                self.tracklets[k] = v
                self.end_time = max(self.end_time, v.detection_history[-1][0])
        return self


class MCTracker:
    # Infinite distance between local tracklets.
    INF = 1

    def __init__(self, stracker_config: Config, captures: Dict[int, Union[ImagesCapture, cv2.VideoCapture]],
                 cluster_freq: int = 60, max_ttl: int = 6000, max_local_overlap: int = 10,
                 max_reid_distance: float = 0.1, n_feature_samples: int = 8):
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
        # Number of feature samples for each new identity
        self.n_feature_samples = n_feature_samples
        # Feature gallery (instead of image gallery)
        self.feature_gallery = np.ndarray((0, 256), dtype=np.float)
        # Corresponding IDs for each line of feature gallery
        self.id_gallery = np.ndarray((0,), dtype=np.int)

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
            # self.update_gallery_features()
            self.query_identities()
        return True

    def update_identity_pool(self):
        """
        Initiate new identities with single-cam tracklets.
        """
        for camID, stracker_state in self.stracker_states.items():
            while len(stracker_state.tracklets_finished) > 0:
                tracklet = stracker_state.tracklets_finished.pop()
                if tracklet.globalID >= 0:
                    identity = self.identity_pool[tracklet.globalID]
                    # TODO: Update Identity features using this tracklet
                else:
                    self.max_id += 1
                    tracklet.globalID = self.max_id
                    self.identity_pool[self.max_id] = Identity(self.max_id, camID, tracklet.id, tracklet,
                                                               n_feature_samples=self.n_feature_samples)

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

    def clear_old_identities(self):
        """
        Clear old identities.
        """
        globalIDs = list(self.identity_pool.keys())
        end_times = np.array([identity.end_time for _, identity in self.identity_pool.items()])
        inds_to_clear = np.where(end_times - self.frame_num < - self.max_ttl)[0]
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
        self.query_identities()

    def query_identities(self):
        if len(self.identity_pool) == 0:
            return
        camIDs = []
        tracklets = []
        M = []
        # Build up query features
        for i, (camID, state) in enumerate(self.stracker_states.items()):
            for tracklet in state.tracklets_active:
                # Only consider tracklets that are long enough
                if len(tracklet.feature_history) >= self.n_feature_samples:
                    camIDs.append(camID)
                    tracklets.append(tracklet)
                    recent_detections = tracklet.detection_history[-len(tracklet.feature_history):]
                    p = np.array([detection.score for _, detection in recent_detections])
                    p = p / np.sum(p)
                    inds = np.random.choice(range(len(tracklet.feature_history)), self.n_feature_samples, replace=False,
                                            p=p)
                    tracklet_features = np.stack([tracklet.feature_history[ind][1]['openreid'].copy() for ind in inds])
                    dists = []
                    for j, (globalID, identity) in enumerate(self.identity_pool.items()):
                        dists.append(np.average(1 - np.matmul(tracklet_features, identity.features.T).reshape(-1)))
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
            print('Local tracklet (cam = {}, localID = {}) matches global ID {} (distance = {})'.format(camIDs[row],
                                                                                                        tracklets[
                                                                                                            row].id,
                                                                                                        identity.globalID,
                                                                                                        M[row][col]))
            tracklets[row].globalID = identity.globalID
            identity.tracklets[(camIDs[row], tracklets[row].id)] = tracklets[row]

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
