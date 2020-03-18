from mot.utils.io import get_capture
from mot.utils.config import file_to_cfg


def test_read_config():
    dict_file_path = 'configs/deepsort.py'
    cfg = file_to_cfg(dict_file_path)
    print(cfg)


def test_image_capture():
    capture = get_capture('/mnt/nasbi/no-backups/datasets/object_tracking/MOTSChallenge/train/images/0002/')
    image = capture.read()
    print(image.shape)


def test_video_capture():
    capture = get_capture('/mnt/nasbi/no-backups/datasets/object_tracking/DukeMTMC/videos/camera1/00000.MTS')
    image = capture.read()
    print(image.shape)


if __name__ == '__main__':
    test_read_config()
    test_image_capture()
    test_video_capture()
