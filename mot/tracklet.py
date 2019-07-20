class Tracklet:
    def __init__(self, id, feature, max_ttl=30):
        self.id = id
        self.feature = feature
        self.max_ttl = max_ttl
        self.ttl = max_ttl
        self.time_lived = 0

    def predict(self):
        raise NotImplementedError('Extend the Tracklet class to implement your own prediction method.')

    def update(self, feature):
        self.feature = feature
        if self.ttl < self.max_ttl:
            self.ttl += 1
        self.time_lived += 1

    def fade(self):
        self.ttl -= 1
        return self.ttl <= 0
