class Tracklet:
    min_time_lived = 3

    def __init__(self, id, frame_id, detection, feature, predictor=None, max_ttl=30, max_feature_history=30, max_detection_history=3000):
        self.id = id
        # Box coordinate of the last target position with (left, top, right, bottom).
        self.last_detection = detection
        # An array storing past features. Only keeping `max_history` frames.
        self.detection_history = [(frame_id, detection)]
        # A feature dictionary.
        self.feature = feature
        # An array storing past features. Only keeping `max_history` frames.
        self.feature_history = [(frame_id, feature)]
        # Max Time-to-Live. Tracklets will get killed if TTL times out.
        self.max_ttl = max_ttl
        # Parameter limiting the past history boxes to keep.
        self.max_detection_history = max_detection_history
        # Parameter limiting the past history features to keep.
        self.max_feature_history = max_feature_history
        # The actual Time-to-Live of a tracklet.
        self.ttl = max_ttl
        # The time lived (time matched with a measurement) of the tracklet.
        self.time_lived = 0
        # The motion predictor, if any.
        self.predictor = predictor
        # Whether the target was just detected or not.
        self.detected = True
        if self.predictor is not None:
            self.predictor.initiate(self)

    def predict(self):
        if self.predictor is not None:
            return self.predictor(self)
        else:
            return self.last_detection

    def update(self, frame_id, detection, feature):
        self.detected = True
        self.last_detection = detection
        self.feature = feature
        if self.predictor is not None:
            self.predictor.predict(self)
            self.predictor.update(self)
        if len(self.feature_history) >= self.max_feature_history:
            self.feature_history.pop(0)
        if len(self.detection_history) >= self.max_detection_history:
            self.detection_history.pop(0)
        self.detection_history.append((frame_id, detection.box))
        self.feature_history.append((frame_id, feature))
        if self.ttl < self.max_ttl:
            self.ttl += 1
        self.time_lived += 1

    def fade(self):
        self.detected = False
        self.last_detection = self.predict()
        self.ttl -= 1
        return self.ttl <= 0

    def is_confirmed(self):
        return self.time_lived > self.min_time_lived

    def is_detected(self):
        return self.detected
