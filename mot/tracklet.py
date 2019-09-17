class Tracklet:
    min_time_lived = 3

    def __init__(self, id, frame_id, box, feature, predictor=None, max_ttl=30, max_history=30, max_box_history=3000):
        self.id = id
        # Box coordinate of the last target position with (left, top, right, bottom).
        self.last_box = box
        # An array storing past features. Only keeping `max_history` frames.
        self.box_history = [(frame_id, box)]
        # A feature dictionary.
        self.feature = feature
        # An array storing past features. Only keeping `max_history` frames.
        self.feature_history = [(frame_id, feature)]
        # Max Time-to-Live. Tracklets will get killed if TTL times out.
        self.max_ttl = max_ttl
        # Parameter limiting the past history boxes to keep.
        self.max_box_history = max_box_history
        # Parameter limiting the past history features to keep.
        self.max_history = max_history
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
            return self.last_box

    def update(self, frame_id, box, feature):
        self.detected = True
        self.last_box = box
        self.feature = feature
        if self.predictor is not None:
            self.predictor.predict(self)
            self.predictor.update(self)
        if len(self.feature_history) >= self.max_history:
            self.feature_history.pop(0)
        if len(self.box_history) >= self.max_box_history:
            self.box_history.pop(0)
        self.box_history.append((frame_id, box))
        self.feature_history.append((frame_id, feature))
        if self.ttl < self.max_ttl:
            self.ttl += 1
        self.time_lived += 1

    def fade(self):
        self.detected = False
        self.last_box = self.predict()
        self.ttl -= 1
        return self.ttl <= 0

    def is_confirmed(self):
        return self.time_lived > self.min_time_lived
