import zmq
import time
import base64
import logging
import traceback
import numpy as np
from threading import Thread, Lock
from typing import List, Tuple, Dict

from mot.utils import Timer

from . import Tracklet, Identity


class ListenerThread(Thread):
    def __init__(self, camID: int, stracker: Dict[str, str], wr_lock: Lock, tracklet_dict: Dict[int, Tracklet],
                 n_feature_samples: int = 8):
        super().__init__()
        self.camID: int = camID
        self.stracker = stracker
        self.stracker['identifier'] = -1
        self.tracker_address = self.stracker['ip'] + ':' + self.stracker['tracker_port']
        self.data_lock: Lock = wr_lock
        self.running: bool = True
        self.tracklet_dict: Dict[int, Tracklet] = tracklet_dict
        self.n_feature_samples: int = n_feature_samples

        context = zmq.Context()
        self.socket = context.socket(zmq.SUB)
        self.socket.connect('tcp://' + self.stracker['ip'] + ':' + self.stracker['tracker_port'])
        self.socket.setsockopt_string(zmq.SUBSCRIBE, np.unicode(''))

    def run(self) -> None:
        while self.running:
            try:
                data = self.socket.recv_string(flags=zmq.NOBLOCK)
                data = base64.b64decode(data)
                data = np.frombuffer(data, dtype=np.float64)
                data = np.reshape(data, (-1, 267))

                if len(data) > 0:
                    identifier = data[0, 0]
                    if self.stracker['identifier'] == -1:
                        self.stracker['identifier'] = identifier
                        logging.getLogger('MTMCT').info(
                            'Identifier of cam server {} is set to {}'.format(self.tracker_address, identifier))
                    else:
                        if self.stracker['identifier'] != identifier:
                            logging.getLogger('MTMCT').warning('Identifier of single-cam tracker invalid')
                            self.stracker['identifier'] = identifier
                            logging.getLogger('MTMCT').warning(
                                'Identifier of cam server {} is set to {}'.format(self.tracker_address, identifier))

                    data = data[:, 1:]
                    logging.getLogger('MTMCT').debug('Received data shape: {} from cam server {}'.format(data.shape,
                                                                                                         self.tracker_address))
                    if self.data_lock.acquire(timeout=500):
                        for line in data:
                            tracklet_id = int(line[1])
                            feature = line[10:]
                            if tracklet_id in self.tracklet_dict:
                                self.tracklet_dict[tracklet_id].add_feature(feature)
                            else:
                                self.tracklet_dict[tracklet_id] = Tracklet(self.camID, tracklet_id, feature,
                                                                           self.n_feature_samples)

                        self.data_lock.release()
            except zmq.ZMQError:
                # print('No data received from IP address ' + self.stracker_ip)
                pass

        if self.data_lock.acquire(timeout=500):
            self.data_lock.release()


class NetworkMCTracker:
    # Infinite distance between local tracklets.
    INF = 1

    def __init__(self,
                 port: int,
                 single_cam_trackers: List[Dict[str, str]],
                 gallery_update_freq: int = 1,
                 min_identity_len: int = 2,
                 max_ttl: int = 6000,
                 max_local_overlap: int = 10,
                 max_reid_distance: float = 0.25,
                 max_cluster_distance: float = 0.1,
                 n_feature_samples: int = 8,
                 min_query: int = 8):
        # Running state
        self.running: bool = True
        # Port for sending global IDs
        self.port = port
        # Single-camera tracker IP addresses
        self.single_cam_trackers: List[Dict[str, str]] = single_cam_trackers
        # Global tracklet pool. globalID => Identity.
        self.identity_pool: Dict[int, Identity] = {}
        # Global ID for allocation
        self.max_id: int = 0
        # Frequency to perform global clustering.
        self.gallery_update_freq: int = gallery_update_freq
        # Timestamp of last gallery update.
        self.last_update_time = time.time()
        # Minimum length of an identity.
        self.min_identity_len = min_identity_len
        # Maximum time-to-live of any tracklet.
        self.max_ttl: int = max_ttl
        # Maximum local overlap time.
        self.max_local_overlap = max_local_overlap
        # Re-identification distance threshold.
        self.max_reid_distance = max_reid_distance
        # Local clustering distance threshold.
        self.max_cluster_distance = max_cluster_distance
        # Number of feature samples for each tracklet.
        self.n_feature_samples = n_feature_samples
        # Minimum size of multi-query.
        self.min_query = min_query
        # Gallery features.
        self.gallery_features = np.zeros((0, 256), dtype=np.float)
        # Corresponding IDs for each line of feature gallery
        self.gallery_ids = np.zeros((0,), dtype=np.int)
        # Logger
        self.logger: logging.Logger = logging.getLogger('MTMCT')
        # Single-camera tracker states
        self.tracklet_dicts: List[Dict[int, Tracklet]] = [dict() for _ in self.single_cam_trackers]
        # Locks for tracklet dicts
        self.data_locks: List[Lock] = [Lock() for _ in self.single_cam_trackers]
        # Listener threads
        self.listener_threads = []

        for i, stracker in enumerate(self.single_cam_trackers):
            listener_thread = ListenerThread(i, stracker, self.data_locks[i], self.tracklet_dicts[i])
            self.listener_threads.append(listener_thread)
            listener_thread.start()

        context = zmq.Context()
        self.sender_socket = context.socket(zmq.PUB)
        self.sender_socket.bind('tcp://*:' + str(self.port))

    def run(self):
        while self.running:
            try:
                # Update identity pool and gallery with a frequency.
                if time.time() - self.last_update_time > self.gallery_update_freq:
                    self.tick()
                    self.last_update_time = time.time()
                    self.log()
            except:
                traceback.print_exc()
                self.terminate()

    @Timer.timer('all')
    def tick(self):
        matches = []
        unmatched = []

        Timer.clear_avg('query')
        for camID, cam_tracklets in enumerate(self.tracklet_dicts):
            self.data_locks[camID].acquire()
            for localID, tracklet in cam_tracklets.items():
                if tracklet.globalID == -1:
                    globalID = self.query(tracklet)
                    if globalID > 0:
                        matches.append((camID, localID, globalID))
                    elif len(tracklet.sample_features()) >= self.min_query:
                        # A tracklet too short is not "unmatched" but "not matched"
                        unmatched.append((camID, localID))
            self.data_locks[camID].release()

        self.update_identity_pool(matches, unmatched)
        self.update_gallery()

    @Timer.timer('idpool_upd')
    def update_identity_pool(self, matches: List[Tuple[int, int, int]], unmatched: List[Tuple[int, int]]):
        for camID, localID, globalID in matches:
            self._update_identity(self.identity_pool[globalID], self.tracklet_dicts[camID][localID])
        for camID, localID in unmatched:
            self._initiate_identity(self.tracklet_dicts[camID][localID])
        self._clear_inactive_identities()

    @Timer.timer('gfpool_upd')
    def update_gallery(self):
        """
        Update identities with new tracklets added, or initiate new identities with single-cam tracklets.
        """
        # TODO: Should we deal with deleted identities? They are not in the identity pool at all.
        gallery_features = []
        gallery_ids = []
        for globalID, identity in self.identity_pool.items():
            features = identity.sample_features()
            gallery_features.append(features)
            gallery_ids.extend([identity.globalID] * len(features))

        if len(gallery_features) > 0:
            self.gallery_features = np.vstack(gallery_features)
            self.gallery_ids = np.array(gallery_ids)

    def _update_identity(self, identity: Identity, tracklet: Tracklet):
        identity.add_tracklet(tracklet)
        tracklet.globalID = identity.globalID

    def _initiate_identity(self, tracklet: Tracklet):
        self.max_id += 1
        self.identity_pool[self.max_id] = Identity(self.max_id, tracklet, self.max_cluster_distance)
        tracklet.globalID = self.max_id
        # Send global ID information to single-cam trackers
        self.sender_socket.send_string('{} {} {}'.format(self.single_cam_trackers[tracklet.camID]['identifier'],
                                                         tracklet.localID, tracklet.globalID))
        self.logger.info('Initiating ID #{} from local target #{} in cam {}'.format(self.max_id, tracklet.localID,
                                                                                    self.single_cam_trackers[
                                                                                        tracklet.camID]['ip'] + ':' +
                                                                                    self.single_cam_trackers[
                                                                                        tracklet.camID][
                                                                                        'tracker_port']))

    @Timer.avg_timer('query')
    def query(self, tracklet: Tracklet) -> int:
        query_features = tracklet.sample_features()
        if len(self.gallery_ids) <= 0 or len(query_features) < self.min_query:
            return -1

        distmat = 1 - np.matmul(query_features, self.gallery_features.T)
        indices = np.argsort(distmat, axis=-1)[:, 0]

        # Filter matches with low similarities
        filtered_inds = np.where(distmat[(np.array(range(len(distmat))), indices)] < self.max_reid_distance)
        indices = indices[filtered_inds]
        if len(indices) == 0:
            return -1

        distance = np.min(distmat[(np.array(range(len(distmat)))[filtered_inds], indices)])
        g_ids = self.gallery_ids[indices]
        matched_g_id = int(np.argmax(np.bincount(g_ids)))

        # Filter matches with time overlaps
        matched_identity = self.identity_pool[matched_g_id]
        if self._time_overlap(tracklet, matched_identity) > self.max_local_overlap:
            self.logger.debug(
                'Tracket {} in cam {} failed to match ID {} due to time overlap'.format(tracklet.localID,
                                                                                        tracklet.camID,
                                                                                        matched_g_id))
            return -1

        # Send matching information to single-cam trackers
        self.sender_socket.send_string('{} {} {}'.format(self.single_cam_trackers[tracklet.camID]['identifier'],
                                                         tracklet.localID, matched_g_id))
        self.logger.info(
            'Tracklet {} in cam {} matched ID {} with min distance {}'.format(tracklet.localID,
                                                                              tracklet.camID,
                                                                              matched_g_id,
                                                                              distance))
        return matched_g_id

    def log(self):
        logstr = '{} features in gallery, {} identities, {} tracklets. '.format(len(self.gallery_ids),
                                                                                len(self.identity_pool),
                                                                                sum(len(d) for d in
                                                                                    self.tracklet_dicts))
        logstr += Timer.logstr()
        self.logger.info(logstr)

    def _clear_inactive_identities(self):
        """
        Clear old inactive identities.
        """
        ids_to_delete = []
        current_time = time.time()
        for globalID, identity in self.identity_pool.items():
            if current_time - identity.last_active_time > self.max_ttl:
                ids_to_delete.append(globalID)
        for globalID in ids_to_delete:
            self.identity_pool.pop(globalID)

        if len(ids_to_delete) > 0:
            self.logger.info('Cleared {} inactive ID(s)'.format(len(ids_to_delete)))

    def terminate(self):
        self.logger.info('Terminating')
        self.running = False
        for thread in self.listener_threads:
            thread.running = False

    def _time_overlap(self, tracklet: Tracklet, identity: Identity):
        overlap = 0
        for _, _tracklet in identity.tracklets.items():
            if tracklet.camID == _tracklet.camID:
                overlap += max(min(tracklet.last_active_time, _tracklet.last_active_time) - \
                               max(tracklet.created_time, _tracklet.created_time),
                               0)
        return overlap

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
