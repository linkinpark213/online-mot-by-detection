import os
import cv2
import argparse
import numpy as np

import mot.utils
import mot.utils.visualize as vis


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("capture_path", type=str)
    parser.add_argument("result_path", type=str)
    parser.add_argument("--ignore_display", dest="display", action="store_false")
    parser.add_argument("--output_path", type=str, default='visualize_videos/', help='Output directory')
    parser.add_argument('--fps', type=int, required=False, default=30, help='Frames per second')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if not os.path.isfile(args.result_path):
        raise FileNotFoundError('Result file path invalid: {}'.format(args.result_path))
    if not os.path.isdir(args.output_path):
        os.makedirs(args.output_path)

    # Read results file
    results = np.loadtxt(args.result_path, delimiter=',')

    # Any capture type
    capture = mot.utils.get_capture(args.capture_path)

    frame_num = 0
    frame_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

    writer_path = os.path.join(args.output_path, args.capture_path.split('/')[-1]) + '.mp4'
    writer = cv2.VideoWriter(writer_path,
                             cv2.VideoWriter_fourcc(*'mp4v'),
                             args.fps,
                             (frame_width, frame_height))
    print('Writing to {}'.format(writer_path))

    while True:
        print('Visualizing frame #{}'.format(frame_num))

        ret, frame = capture.read()
        if not ret:
            break

        # Draw all target boxes in the frame
        boxes = results[np.where(results[:, 0] == frame_num)]

        frame = vis.snapshot_from_results(frame, boxes, frame_num)

        # Write (and show)
        writer.write(frame)

        if args.display:
            cv2.imshow('Frame', frame)
            cv2.waitKey(1)

        frame_num += 1

    if writer is not None:
        writer.release()
