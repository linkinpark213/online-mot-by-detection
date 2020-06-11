import cv2
import numpy as np
import scipy.cluster.hierarchy as sch
from typing import List, Union, Tuple, Dict

from mot.utils.config import Config
from mot.structures import Tracklet
from mot.utils.io import ImagesCapture
from mot.tracker import build_tracker, Tracker, TrackerState


class Identity:
    def __init__(self, camID: int, localID: int, tracklet: Tracklet):
        self.end_time: int = tracklet.detection_history[-1][0]
        self.tracklets: Dict[Tuple[int, int], Tracklet] = {(camID, localID): tracklet}
        # Only use the feature of the most confident detection
        recent_detections = tracklet.detection_history[-len(tracklet.feature_history):]
        best_ind = int(np.argmax([detection.score for _, detection in recent_detections]))
        tracklet.feature = tracklet.feature_history[best_ind][1]
        del tracklet.feature_history

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
                 max_reid_distance: float = 0.1):
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

    def tick(self):
        self.frame_num += 1
        for camID, cap in self.captures.items():

            ret, frame = cap.read()
            if not ret:
                return False
            self.stracker.state = self.stracker_states[camID]
            self.stracker.tick(frame)

        if self.frame_num % self.cluster_freq == 0:
            self.initiate_new_identities()
            self.hierarchical_cluster()
        return True

    def initiate_new_identities(self):
        # Initiate new identities with single-cam tracklets.
        for camID, stracker_state in self.stracker_states.items():
            while len(stracker_state.tracklets_finished) > 0:
                tracklet = stracker_state.tracklets_finished.pop()
                self.max_id += 1
                self.identity_pool[self.max_id] = Identity(camID, tracklet.id, tracklet)

    def clear_old_identities(self):
        # Clear old identities.
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
        self.initiate_new_identities()
        self.hierarchical_cluster()

    def hierarchical_cluster(self):
        N = len(self.identity_pool)
        if N <= 1:
            return
        M = np.zeros([N, N])
        globalIDs = list(self.identity_pool.keys())
        for i in range(N):
            for j in range(i + 1, N):
                M[i][j] = self._identity_distance(self.identity_pool[globalIDs[i]], self.identity_pool[globalIDs[j]])

        print('Affinity matrix')
        for line in M:
            print(line)

        iu = np.triu_indices(N, 1, N)
        D = M[iu]
        Z = sch.linkage(D, method='average')
        clusterIDs = sch.fcluster(Z, self.max_reid_distance, criterion='distance')
        unique_clusterIDs = np.unique(clusterIDs)
        for clusterID in unique_clusterIDs:
            inds = np.where(clusterIDs == clusterID)[0]
            if len(inds) > 1:
                print('Clustering: ')
                for i, ind in enumerate(inds):
                    identity = self.identity_pool[globalIDs[ind]]
                    print('\tGlobal identity #{}'.format(globalIDs[ind]), end=' ')
                    if i == 0:
                        print('(#1 in cluster)')
                    else:
                        print('(distance to #1 in cluster = {})'.format(M[inds[0]][ind]))
                    for camID, localID in identity.tracklets.keys():
                        print('\t\t(Cam #{}, target #{})'.format(camID, localID))
                self.identity_pool[globalIDs[inds[0]]].merge([self.identity_pool[globalIDs[ind]] for ind in inds[1:]])
                for ind in inds[1:]:
                    self.identity_pool.pop(globalIDs[ind])

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
