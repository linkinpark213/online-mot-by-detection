import os
import cv2


class ImagesCapture:
    def __init__(self, images_path):
        self.images_path = images_path
        self.n = 0
        self.image_filenames = os.listdir(images_path)
        self.image_filenames.sort()

    def read(self):
        if self.n >= len(self.image_filenames):
            return False, None
        image = cv2.imread(os.path.join(self.images_path, self.image_filenames[self.n]))
        self.n += 1
        if image is not None:
            return True, image
        return False, None