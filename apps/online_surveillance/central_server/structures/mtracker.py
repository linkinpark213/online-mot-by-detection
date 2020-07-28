import zmq
import time
import base64
import logging
import numpy as np
from threading import Thread, Lock
from typing import List, Tuple, Dict

from mot.utils import Timer

from . import Tracklet, Identity


class ListenerThread(Thread):
    def __init__(self, camID: int, stracker: Dict[str, str], wr_lock: Lock, tracklet_dict: Dict[int, Tracklet],
                 n_feature_samples: int = 8):
        super().__init__()
        # Integer camera ID.
        self.camID: int = camID
        # Single-camera tracker configs.
        #       'identifier'    => int
        #       'ip'            => str
        #       'footage_port'  => str
        #       'tracker_port'  => str
        #       'listen_port'   => str
        self.stracker: Dict[str, str] = stracker
        self.stracker['identifier']: int = -1
        # Tracker's "IP address : port".
        self.tracker_address: str = self.stracker['ip'] + ':' + self.stracker['tracker_port']
        # Lock for synchronization.
        self.data_lock: Lock = wr_lock
        # Running state.
        self.running: bool = True
        # Dict for storing tracklets. Require synchronization when reading/writing.
        self.tracklet_dict: Dict[int, Tracklet] = tracklet_dict
        # Number of sampled features for each tracklet.
        self.n_feature_samples: int = n_feature_samples
        # ZeroMQ context and sockets for broadcasting global querying results.
        context = zmq.Context()
        self.socket = context.socket(zmq.SUB)
        self.socket.connect('tcp://' + self.stracker['ip'] + ':' + self.stracker['tracker_port'])
        self.socket.setsockopt_string(zmq.SUBSCRIBE, np.unicode(''))

    def run(self) -> None:
        while self.running:
            try:
                # Receive data from remote single–camera server.
                # 267 = 1 (identifier) + 10 numbers (MOT-format data) + 256 (feature dimension).
                data = self.socket.recv_string(flags=zmq.NOBLOCK)
                data = base64.b64decode(data)
                data = np.frombuffer(data, dtype=np.float64)
                data = np.reshape(data, (-1, 267))

                if len(data) > 0:
                    # All identifiers are the same, indicating which camera it is.
                    # Re-ID matches are broadcast, and cameras have to know which match are local tracklets in it.
                    identifier = data[0, 0]
                    if self.stracker['identifier'] == -1:
                        # The first time receiving data from a cam server, initialize its identifier.
                        self.stracker['identifier'] = identifier
                        logging.getLogger('MTMCT').info(
                            'Identifier of cam server {} is set to {}'.format(self.tracker_address, identifier))
                    else:
                        # Not the first time, validate identifier.
                        # If not equal, we modify the saved identifier.
                        if self.stracker['identifier'] != identifier:
                            logging.getLogger('MTMCT').warning('Identifier of single-cam tracker invalid')
                            self.stracker['identifier'] = identifier
                            logging.getLogger('MTMCT').warning(
                                'Identifier of cam server {} is set to {}'.format(self.tracker_address, identifier))

                    # Remove the identifier column, leaving (N, 266)
                    data = data[:, 1:]
                    logging.getLogger('MTMCT').debug('Received data shape: {} from cam server {}'.format(data.shape,
                                                                                                         self.tracker_address))

                    # Synchronized - When updating, avoid reading.
                    self.data_lock.acquire()
                    for line in data:
                        tracklet_id = int(line[1])
                        feature = line[10:]
                        if tracklet_id in self.tracklet_dict:
                            # Old tracklet, append feature to it.
                            self.tracklet_dict[tracklet_id].add_feature(feature)
                        else:
                            # New tracklet, initialize.
                            self.tracklet_dict[tracklet_id] = Tracklet(self.camID, tracklet_id, feature,
                                                                       self.n_feature_samples)

                    self.data_lock.release()
            except zmq.ZMQError:
                # print('No data received from IP address ' + self.stracker_ip)
                pass


class NetworkMCTracker:
    # Infinite distance between local tracklets.
    INF = 1

    def __init__(self,
                 port: int,
                 single_cam_trackers: List[Dict[str, str]],
                 gallery_update_freq: int = 1,
                 max_ttl: int = 6000,
                 max_local_overlap: int = 10,
                 max_reid_distance: float = 0.25,
                 max_cluster_distance: float = 0.1,
                 n_feature_samples: int = 8,
                 min_query: int = 8,
                 **kwargs):
        # Running state.
        self.running: bool = True
        # Port for sending global IDs.
        self.port = port
        # Single-camera tracker IP addresses.
        self.single_cam_trackers: List[Dict[str, str]] = single_cam_trackers
        # Global tracklet pool. globalID => Identity.
        self.identity_pool: Dict[int, Identity] = {}
        # Global ID for allocation.
        self.max_id: int = 0
        # Frequency to perform global clustering.
        self.gallery_update_freq: int = gallery_update_freq
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
        # Matches, a list of (camID, localID, globalID, distance) tuples
        self.matches: List[Tuple[int, int, int, float]] = []
        # Unmatched tracklets, a list of (camID, localID) tuples
        self.unmatched_tracklets: List[Tuple[int, int, int]] = []

        for i, stracker in enumerate(self.single_cam_trackers):
            listener_thread = ListenerThread(i, stracker, self.data_locks[i], self.tracklet_dicts[i])
            self.listener_threads.append(listener_thread)
            listener_thread.start()

        context = zmq.Context()
        self.sender_socket = context.socket(zmq.PUB)
        self.sender_socket.bind('tcp://*:' + str(self.port))

    @Timer.timer('all')
    def tick(self):
        # Perform query-search.
        self.matches = []
        self.unmatched_tracklets = []

        Timer.clear_avg('query')
        for camID, cam_tracklets in enumerate(self.tracklet_dicts):
            # Synchronized - When querying, avoid listener change tracklet dicts
            self.data_locks[camID].acquire()
            for localID, tracklet in cam_tracklets.items():
                if tracklet.globalID == -1:
                    globalID, distance = self.query(tracklet)
                    if globalID > 0:
                        # Successfully matched to an identity.
                        self.matches.append((camID, localID, globalID, distance))
                    elif len(tracklet.sample_features()) >= self.min_query:
                        # A tracklet too short is not "unmatched" but "not matched".
                        # So only tracklets with enough features will be initialized.
                        self.unmatched_tracklets.append((camID, localID, -1))
            self.data_locks[camID].release()

        # Update identity pools using matched / unmatched tracklet IDs.
        self.update_identity_pool(self.matches, self.unmatched_tracklets)

        # Update gallery features and corresponding IDs using updated identity pool.
        self.update_gallery()

    @Timer.timer('idpool_upd')
    def update_identity_pool(self, matches: List[Tuple[int, int, int, float]], unmatched: List[Tuple[int, int, int]]):
        # Add matched tracklets to identities.
        for camID, localID, globalID, distance in matches:
            self._update_identity(self.identity_pool[globalID], self.tracklet_dicts[camID][localID])

        # Initiate identities for unmatched tracklets.
        n_unmatched = len(unmatched)
        for i in range(n_unmatched):
            camID, localID, _ = unmatched.pop(0)
            globalID = self._initiate_identity(self.tracklet_dicts[camID][localID])
            unmatched.append((camID, localID, globalID))

        # Remove inactive identities from identity pool, and remove their tracklets from tracklet dict.
        self._clear_inactive_identities()
        # Remove unmatched tracklets from identity pool
        self._clear_inactive_tracklets()

    @Timer.timer('gfpool_upd')
    def update_gallery(self):
        """
        Update identities with new tracklets added, or initiate new identities with single-cam tracklets.
        """
        gallery_features = []
        gallery_ids = []
        for globalID, identity in self.identity_pool.items():
            features = identity.sample_features()
            gallery_features.append(features)
            gallery_ids.extend([identity.globalID] * len(features))

        if len(gallery_features) > 0:
            self.gallery_features = np.vstack(gallery_features)
            self.gallery_ids = np.array(gallery_ids)
        else:
            self.gallery_features = np.zeros((0, 256), dtype=np.float)
            self.gallery_ids = np.zeros((0,), dtype=np.int)

    @Timer.avg_timer('query')
    def query(self, tracklet: Tracklet) -> Tuple[int, float]:
        query_features = tracklet.sample_features()
        if len(self.gallery_ids) <= 0 or len(query_features) < self.min_query:
            return -1, 0

        distmat = 1 - np.matmul(query_features, self.gallery_features.T)
        g_indices = np.argsort(distmat, axis=-1)[:, 0]

        # Filter matches with low similarities.
        indices_filter = np.where(distmat[(np.array(range(len(distmat))), g_indices)] < self.max_reid_distance)[0]
        if len(indices_filter) == 0:
            return -1, np.min(distmat[(np.array(range(len(distmat))), g_indices)])

        # Remove low-similarity indices of matches
        g_indices = g_indices[indices_filter]
        q_indices = np.array(range(len(distmat)))[indices_filter]

        min_distance = np.min(distmat[(q_indices, g_indices)])

        g_ids = self.gallery_ids[g_indices]
        matched_g_id = int(np.argmax(np.bincount(g_ids)))

        # Filter matches with time overlaps
        matched_identity = self.identity_pool[matched_g_id]
        if self._time_overlap(tracklet, matched_identity) > self.max_local_overlap:
            self.logger.debug(
                'Tracket {} in cam {} failed to match ID {} due to time overlap'.format(tracklet.localID,
                                                                                        tracklet.camID,
                                                                                        matched_g_id))
            return -1, 0

        # Send matching information to single-cam trackers
        self.sender_socket.send_string('{} {} {}'.format(self.single_cam_trackers[tracklet.camID]['identifier'],
                                                         tracklet.localID, matched_g_id))
        self.logger.info(
            'Tracklet {} in cam {} matched ID {} with min distance {}'.format(tracklet.localID,
                                                                              tracklet.camID,
                                                                              matched_g_id,
                                                                              min_distance))
        return matched_g_id, min_distance

    def log(self):
        """
        Print a log string using logging.
        """
        tracklet_nums = '('
        for d in self.tracklet_dicts:
            tracklet_nums += ' ' + str(len(d)) + ' + '
        tracklet_nums = tracklet_nums[:-2] + ')'

        logstr = '{} features in gallery, {} identities, {} tracklets. '.format(len(self.gallery_ids),
                                                                                len(self.identity_pool),
                                                                                tracklet_nums)
        logstr += Timer.logstr()
        self.logger.info(logstr)

    def terminate(self):
        """
        Terminate multi-camera tracker.
        """
        self.logger.info('Terminating')
        self.running = False
        for thread in self.listener_threads:
            thread.running = False

    def _update_identity(self, identity: Identity, tracklet: Tracklet):
        """
        Update an identity, adding a new tracklet matched to it.

        Args:
            identity: The matched identity.
            tracklet: The matched tracklet.
        """
        identity.add_tracklet(tracklet)
        tracklet.globalID = identity.globalID

    def _initiate_identity(self, tracklet: Tracklet):
        """
        Initiate a new identity from a tracklet.

        Args:
            tracklet: A tracklet object.

        Returns:
            An integer. The generated global ID for the new identity.
        """
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
        return self.max_id

    def _clear_inactive_identities(self):
        """
        Clear old inactive identities.
        """
        # Keep record of global IDs of too-old identities.
        ids_to_delete = []
        current_time = time.time()
        for globalID, identity in self.identity_pool.items():
            if current_time - identity.last_active_time > self.max_ttl:
                ids_to_delete.append(globalID)

        # Remove after traversal.
        for globalID in ids_to_delete:
            # Both identity pool,
            identity = self.identity_pool.pop(globalID)
            # And tracklet dicts.
            for camID, localID in identity.tracklets.keys():
                self.data_locks[camID].acquire()
                self.tracklet_dicts[camID].pop(localID)
                self.data_locks[camID].release()

        if len(ids_to_delete) > 0:
            self.logger.info('Cleared {} inactive ID(s): {}'.format(len(ids_to_delete), ids_to_delete))

    def _clear_inactive_tracklets(self):
        """
        Clear old inactive tracklets.
        """
        for cam, d in enumerate(self.tracklet_dicts):
            ids_to_delete = []
            current_time = time.time()
            self.data_locks[cam].acquire()
            for localID, tracklet in d.items():
                if tracklet.globalID == -1 and current_time - tracklet.last_active_time > self.max_ttl:
                    ids_to_delete.append(localID)

            for localID in ids_to_delete:
                d.pop(localID)

            if len(ids_to_delete) > 0:
                self.logger.info('Cleared {} inactive tracklet(s) in cam {}'.format(len(ids_to_delete), cam))

            self.data_locks[cam].release()

    def _time_overlap(self, tracklet: Tracklet, identity: Identity):
        overlap = 0
        for _, _tracklet in identity.tracklets.items():
            if tracklet.camID == _tracklet.camID:
                overlap += max(min(tracklet.last_active_time, _tracklet.last_active_time) - \
                               max(tracklet.created_time, _tracklet.created_time),
                               0)
        return overlap
