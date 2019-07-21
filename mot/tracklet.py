class Tracklet:
    def __init__(self, id, feature, predictor=None, max_ttl=30, max_history=30):
        self.id = id
        self.feature = feature
        self.history = []
        self.max_ttl = max_ttl
        self.max_history = max_history
        self.ttl = max_ttl
        self.time_lived = 0
        self.predictor = predictor

    def predict(self):
        if self.predictor is not None:
            return self.predictor(self.history)
        else:
            return self.feature

    def update(self, n, feature):
        self.feature = feature
        if len(self.history) >= self.max_history:
            self.history.pop(0)
        self.history.append((n, feature))
        if self.ttl < self.max_ttl:
            self.ttl += 1
        self.time_lived += 1

    def fade(self):
        self.ttl -= 1
        return self.ttl <= 0
