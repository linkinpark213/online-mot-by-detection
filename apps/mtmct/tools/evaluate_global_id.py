import os
import re
import logging
import argparse
import numpy as np
import scipy.io as sio
from typing import Dict, Tuple
from scipy.optimize import linear_sum_assignment

from mot.utils import iou

VIDEO_START_TIMES = np.array([0, 5543, 3607, 27244, 31182, 1, 22402, 18968, 46766])
DATA_START_FRAMES = np.array([0, 44158, 46094, 26078, 18913, 50742, 27476, 30733, 2935])


def parse_args():
    parser = argparse.ArgumentParser('Global ID evaluation')
    parser.add_argument('--gt-path', type=str, required=True, help='Path to the ground truth .mat file')
    parser.add_argument('--result-path', type=str, required=True, help='Directory of the tracking results')
    parser.add_argument('--start-time', type=int, required=True, help='Global frame number of starting point')
    parser.add_argument('--fps', type=int, required=False, default=60, help='FPS of tracking results, default is 60')
    parser.add_argument('--match-thresh', type=float, required=False, default=0.5,
                        help='Threshold for matching gt box with prediction box')
    args = parser.parse_args()

    assert args.start_time >= 53321, 'Starting time should be no lower than 53321.'
    assert args.fps <= 60, 'FPS should not be higher than 60 (original video frame rate)'

    pattern = re.compile('out_[1-8].txt')
    cams = []
    for filename in os.listdir(args.result_path):
        if pattern.fullmatch(filename):
            cams.append(int(filename.split('.')[0].split('_')[1]))
        else:
            assert filename == 'out_mc.txt', 'Expected only out_${cam}.txt or out_mc.txt in result path'
    args.cams = cams
    args.start_frames = args.start_time - VIDEO_START_TIMES
    args.sample_rate = 60 / args.fps

    return args


def load_data(args):
    max_frames = 0
    local_results = {}
    for cam in args.cams:
        local_result = np.loadtxt(os.path.join(args.result_path, 'out_{}.txt'.format(cam)), delimiter=',')
        logging.info('Cam #{}: Local result shape: {}'.format(cam, local_result.shape))
        logging.info('Cam #{}: Number of local tracklets: {}'.format(cam, len(np.unique(local_result[:, 1]))))
        max_frames = max(max_frames, int(local_result[:, 0].max()))
        local_results[cam] = local_result
        # To (l, t, r, b)
        local_result[:, 4] = local_result[:, 4] + local_result[:, 2]
        local_result[:, 5] = local_result[:, 5] + local_result[:, 3]

    global_result = np.loadtxt(os.path.join(args.result_path, 'out_mc.txt'), dtype=np.int)
    logging.info('Global result shape: {}'.format(global_result.shape))
    logging.info('Number of global IDs: {}'.format(len(np.unique(global_result[:, 0]))))
    local_2_global = {}
    for globalID, camID, localID in global_result:
        local_2_global[(camID, localID)] = globalID

    trainData = sio.loadmat(args.gt_path)['trainData']
    logging.info('All train data shape: {}'.format(trainData.shape))

    gts = []
    for cam in args.cams:
        gt = trainData[np.where(trainData[:, 0] == cam)]
        start_frame = args.start_time - VIDEO_START_TIMES[cam]
        gt = gt[np.where(np.logical_and(gt[:, 2] >= start_frame,
                                        gt[:, 2] <= start_frame + args.sample_rate * max_frames))]
        frame_ids = (start_frame + np.arange(0, args.sample_rate * max_frames, args.sample_rate)).astype(np.int)

        gt = gt[np.where(np.in1d(gt[:, 2], frame_ids))]
        gt[:, 2] = 1 + (gt[:, 2] - start_frame) / args.sample_rate

        gts.append(gt)

    gts = np.vstack(gts)
    # To (l, t, r, b)
    gts[:, 5] = gts[:, 5] + gts[:, 3]
    gts[:, 6] = gts[:, 6] + gts[:, 4]

    logging.info('Valid train data shape: {}'.format(gts.shape))
    logging.info('Number of ground truth IDs: {}'.format(len(np.unique(gts[:, 1]))))

    cross_cam_ids = set()
    for gt_id in np.unique(gts[:, 1]):
        if len(np.unique(gts[np.where(gts[:, 1] == gt_id)][:, 0])) > 1:
            cross_cam_ids.add(int(gt_id))

    logging.info('Number if cross-cam ground truth IDs: {}'.format(len(cross_cam_ids)))
    return gts, local_results, local_2_global


def evaluate(args, gts: np.ndarray, local_results: Dict[int, np.ndarray], local_2_global: Dict[Tuple[int, int], int]):
    FN, FP = [], []
    association = {}
    for cam in args.cams:
        cam_gt = gts[np.where(gts[:, 0] == cam)]
        pred = local_results[cam]
        for frame_id in np.unique(cam_gt[:, 2]):
            frame_id = int(frame_id)
            frame_gt = cam_gt[np.where(cam_gt[:, 2] == frame_id)]
            frame_pred = pred[np.where(pred[:, 0] == frame_id)]
            if len(frame_gt) > 0:
                if len(frame_pred) > 0:
                    iou_matrix = []
                    for gt_box in frame_gt:
                        iou_matrix.append(iou(gt_box[3: 7], frame_pred[:, 2:6]))
                    iou_matrix = np.array(iou_matrix)
                    row_inds, col_inds = linear_sum_assignment(1 - iou_matrix)
                    valid_inds = [iou_matrix[row_inds[i], col_inds[i]] > args.match_thresh for i in
                                  range(len(row_inds))]
                    row_inds, col_inds = row_inds[valid_inds], col_inds[valid_inds]

                    # Matches
                    for row_ind, col_ind in zip(row_inds, col_inds):
                        gt_id = int(frame_gt[row_ind, 1])
                        local_pred_id = int(frame_pred[col_ind, 1])
                        if gt_id not in association.keys():
                            association[gt_id] = {}
                        if (cam, local_pred_id) in local_2_global:
                            association[gt_id][(cam, frame_id)] = local_2_global[(cam, local_pred_id)]
                        else:
                            logging.debug('Local ID #{} in cam #{} not in global results'.format(local_pred_id, cam))

                    # FN
                    for ind in (set(range(len(frame_gt))) - set(row_inds)):
                        gt_box = frame_gt[ind]
                        cam, gt_id, frame_id = map(int, gt_box[:3])
                        FN.append((cam, frame_id, gt_id))
                        logging.debug('FN: Cam #{}, GT ID #{}, Frame #{}'.format(cam, gt_id, frame_id))

                    # FP
                    for ind in (set(range(len(frame_pred))) - set(col_inds)):
                        pred_box = frame_pred[ind]
                        frame_id, pred_id = map(int, pred_box[:2])
                        FP.append((cam, frame_id, pred_id))
                        logging.debug('FP: Cam #{}, pred ID #{}, Frame #{}'.format(cam, pred_id, frame_id))
                else:
                    # No detections, add to FNs
                    for gt_box in frame_gt:
                        cam, gt_id, frame_id = map(int, gt_box[:3])
                        FN.append((cam, frame_id, gt_id))
                        logging.debug('FN: Cam #{}, GT ID #{}, Frame #{}'.format(cam, gt_id, frame_id))

    TR, FR = 0, 0
    for gt_id in association.keys():
        related_cams = set()
        all_matched_pred_ids = []
        for (cam, frame_id), pred_id in association[gt_id].items():
            related_cams.add(cam)
            if len(all_matched_pred_ids) > 0 and pred_id not in all_matched_pred_ids:
                logging.debug('FR: GT #{} ({}) unrecognized at c{}, f{}, but has predicted ID #{}'.format(gt_id,
                                                                                                          all_matched_pred_ids[
                                                                                                              0],
                                                                                                          cam,
                                                                                                          frame_id,
                                                                                                          pred_id))
                all_matched_pred_ids.append(pred_id)
            elif len(all_matched_pred_ids) == 0:
                all_matched_pred_ids.append(pred_id)

        if len(related_cams) > 1:
            FR += len(all_matched_pred_ids) - 1
            TR += len(related_cams) - 1

    logging.info('FN = {}, FP = {}'.format(len(FN), len(FP)))
    logging.info('Failed recognitions = {}'.format(FR))
    logging.info('Successful recognitions = {}'.format(TR))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    args = parse_args()

    gt, local_results, local_2_global = load_data(args)

    evaluate(args, gt, local_results, local_2_global)
