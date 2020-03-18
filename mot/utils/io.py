import os
import cv2

__all__ = ['get_capture', 'get_result_writer', 'get_video_writer']


class ImagesCapture:
    def __init__(self, images_path):
        self.images_path = images_path
        self.n = 0
        self.image_filenames = os.listdir(images_path)
        self.image_filenames.sort()
        if self.image_filenames[0][0] == '.':
            self.image_filenames.pop(0)

    def read(self):
        if self.n >= len(self.image_filenames):
            return False, None
        image = cv2.imread(os.path.join(self.images_path, self.image_filenames[self.n]))
        assert image is not None, 'Image file {} is not valid'.format(os.path.join(self.images_path,
                                                                                   self.image_filenames[self.n]))
        self.n += 1
        return True, image

    def get(self, propId):
        if propId == cv2.CAP_PROP_FRAME_COUNT:
            return len(self.image_filenames)
        elif propId == cv2.CAP_PROP_FRAME_HEIGHT:
            return cv2.imread(os.path.join(self.images_path, self.image_filenames[0])).shape[0]
        elif propId == cv2.CAP_PROP_FRAME_WIDTH:
            return cv2.imread(os.path.join(self.images_path, self.image_filenames[0])).shape[1]


class DummyWriter():
    def write(self, *args, **kwargs):
        pass

    def close(self):
        pass

    def release(self):
        pass


def get_capture(demo_path):
    if demo_path == '':
        return cv2.VideoCapture(0)
    else:
        if os.path.isdir(demo_path):
            return ImagesCapture(demo_path)
        elif os.path.isfile(demo_path):
            return cv2.VideoCapture(demo_path)
        else:
            raise AssertionError('Parameter "demo_path" is not a file or directory.')


def get_video_writer(save_video_path, width, height):
    if save_video_path != '':
        return cv2.VideoWriter(save_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (int(width), int(height)))
    else:
        return DummyWriter()


def get_result_writer(save_result_path):
    if save_result_path != '':
        return open(save_result_path, 'w+')
    else:
        return DummyWriter()
