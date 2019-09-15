import os
import cv2
import argparse
import numpy as np
import mot.utils.vis as vis


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", type=str)
    parser.add_argument("result_path", type=str)
    parser.add_argument("--ignore_display", dest="display", action="store_false")
    parser.add_argument("--output_path", type=str, default='videos/')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if not os.path.isfile(args.video_path):
        raise FileNotFoundError('Video file path invalid: {}'.format(args.video_path))
    if not os.path.isfile(args.result_path):
        raise FileNotFoundError('Result file path invalid: {}'.format(args.result_path))
    if not os.path.isdir(args.output_path):
        os.makedirs(args.output_path)

    results = np.loadtxt(args.result_path, delimiter=',')
    results[:, 4] = results[:, 4] + results[:, 2]
    results[:, 5] = results[:, 5] + results[:, 3]
    capture = cv2.VideoCapture(args.video_path)
    writer = None
    frame_no = 0

    while True:
        ret, frame = capture.read()
        if not ret:
            break
        if writer is None:
            writer = cv2.VideoWriter(os.path.join(args.output_path, args.video_path.split('/')[-1]),
                                     cv2.VideoWriter_fourcc(*'mp4v'),
                                     30, (frame.shape[1], frame.shape[0]))

        boxes = results[np.where(results[:, 0] == frame_no)]
        for box in boxes:
            frame = vis.draw_object(frame, box[2:6], box[1])

        writer.write(frame)
        if args.display:
            cv2.imshow('Frame', frame)
            cv2.waitKey(1)

        frame_no += 1

    if writer is not None:
        writer.release()
