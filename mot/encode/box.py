from .encode import Encoder


class BoxCoordinateEncoder(Encoder):
    def __init__(self):
        super(BoxCoordinateEncoder).__init__()

    def __call__(self, boxes, img):
        return boxes
