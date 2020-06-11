import cv2
import argparse
import numpy as np

from mot.utils import get_capture, snapshot_from_results, get_video_writer


def generate_global_dict(global_results, camID):
    print(global_results.shape)
    global_results = global_results[np.where(global_results[:, 1] == camID)]
    d = {}
    for globalID, camID, localID in global_results:
        d[localID] = globalID
    print('All local IDs:')
    print(d.keys())
    return d


def to_global_id(local_results, global_dict):
    for i in range(len(local_results)):
        localID = int(local_results[i, 1])
        if localID in global_dict:
            local_results[i, 1] = global_dict[localID]
        else:
            local_results[i, 1] = -1
            print('Local ID {} not in global identities!'.format(localID))
    return local_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('capture_path', type=str)
    parser.add_argument('cam_id', type=int)
    parser.add_argument('global_result_path', type=str)
    parser.add_argument('local_result_path', type=str)
    args = parser.parse_args()

    capture = get_capture(args.capture_path)
    camID = args.cam_id
    global_results = np.loadtxt(args.global_result_path)
    local_results = np.loadtxt(args.local_result_path, delimiter=',')

    global_dict = generate_global_dict(global_results, camID)
    local_results = to_global_id(local_results, global_dict)

    writer = get_video_writer('out_global_{}.mp4'.format(args.cam_id), 1920, 1080, fps=60)

    for frame_num in range(int(min(local_results[:, 0])), int(max(local_results[:, 0]) + 1)):
        boxes = local_results[np.where(local_results[:, 0] == frame_num)]
        print('{} detections in Frame {}'.format(len(boxes), frame_num))
        ret, frame = capture.read()

        writer.write(snapshot_from_results(frame, boxes, frame_num + 1))

    writer.release()
