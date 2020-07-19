import os
import cv2
import time
import logging
import threading
import numpy as np
from abc import abstractmethod, ABCMeta

__all__ = ['Capture', 'get_capture', 'get_result_writer', 'get_video_writer', 'RealTimeWrapper']


class Capture(metaclass=ABCMeta):
    @abstractmethod
    def read(self):
        pass

    @abstractmethod
    def get(self, propId):
        pass

    def skip(self, n):
        for i in range(n):
            self.read()

    def release(self):
        pass


class ImagesCapture(Capture):
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

    def skip(self, n):
        self.n += n

    def get(self, propId):
        if propId == cv2.CAP_PROP_FRAME_COUNT:
            return len(self.image_filenames)
        elif propId == cv2.CAP_PROP_FRAME_HEIGHT:
            return cv2.imread(os.path.join(self.images_path, self.image_filenames[0])).shape[0]
        elif propId == cv2.CAP_PROP_FRAME_WIDTH:
            return cv2.imread(os.path.join(self.images_path, self.image_filenames[0])).shape[1]


class VideoCapture(Capture):
    def __init__(self, capture: cv2.VideoCapture):
        self.capture = capture

    def read(self):
        return self.capture.read()

    def get(self, propId):
        return self.capture.get(propId)


class _RealTimeCaptureThread(threading.Thread):
    def __init__(self, wrapper, capture, original_fps: int, lock: threading.Lock, start_time: float = 0):
        super().__init__()
        self.wrapper = wrapper
        self.capture = capture
        self.original_fps = original_fps
        self.lock = lock
        self.wait_time = 1. / self.original_fps
        self.start_time = start_time if start_time > 0 else time.time()
        self.running = True
        self.n = 0

    def run(self) -> None:
        while self.running:
            current_time = time.time()
            if current_time >= self.start_time + self.wait_time * self.n:
                # To fake a real-time frame rate, the expected frame number should be calculated every time
                expected_n = int((current_time - self.start_time) / self.wait_time)
                if expected_n > self.n + 1:
                    logging.getLogger('MOT').debug(
                        'Skipping by {} frames to {}'.format(expected_n - self.n - 1, expected_n))
                    self.capture.skip(expected_n - self.n - 1)
                    self.n = expected_n - 1

                self.n += 1
                ret, frame = self.capture.read()
                if not self.lock.acquire(timeout=1):
                    self.running = False
                    raise RuntimeError('Real-time capture thread can\'t acquire lock')
                self.wrapper.ret = ret
                self.wrapper.frame = frame
                self.lock.release()


class RealTimeWrapper(Capture):
    def __init__(self, capture, original_fps: int):
        self.ret = False
        self.frame = None
        self.capture = capture
        self.lock = threading.Lock()
        self.capture_thread = _RealTimeCaptureThread(self, capture, original_fps, self.lock, time.time())
        self.ret, self.frame = self.capture.read()
        self.started = False

    def read(self):
        if not self.started:
            self.capture_thread.start()
            self.started = True

        if not self.lock.acquire(timeout=1):
            return False, None
        ret, frame = self.ret, None if not self.ret else self.frame.copy()
        self.lock.release()
        return ret, frame

    def get(self, propId):
        return self.capture.get(propId)

    def skip(self, n):
        self.capture.skip(n)

    def release(self):
        self.capture_thread.running = False


class DummyWriter():
    def write(self, *args, **kwargs):
        pass

    def close(self):
        pass

    def release(self):
        pass


def get_capture(demo_path: str) -> Capture:
    if not os.path.exists(demo_path):
        try:
            capture = VideoCapture(cv2.VideoCapture(int(demo_path)))
            return capture
        except ValueError:
            raise AssertionError('Parameter "demo_path" is not a file or directory.')
    else:
        if os.path.isdir(demo_path):
            return ImagesCapture(demo_path)
        elif os.path.isfile(demo_path):
            return VideoCapture(cv2.VideoCapture(demo_path))
        else:
            raise AssertionError('Parameter "demo_path" is not a file or directory.')


def get_video_writer(save_video_path, width, height, fps: int = 30):
    if save_video_path != '':
        save_video_dir = os.path.dirname(os.path.abspath(save_video_path))
        if not os.path.isdir(save_video_dir):
            logging.getLogger('MOT').warning('Video saving path {} doens\'t exist. Creating...')
            os.makedirs(save_video_dir)
        return cv2.VideoWriter(save_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(width), int(height)))
    else:
        return DummyWriter()


def get_result_writer(save_result_path):
    if save_result_path != '':
        save_result_dir = os.path.dirname(os.path.abspath(save_result_path))
        if not os.path.isdir(save_result_dir):
            logging.getLogger('MOT').warning('Result saving path {} doens\'t exist. Creating...')
            os.makedirs(save_result_dir)
        return open(save_result_path, 'w+')
    else:
        return DummyWriter()
