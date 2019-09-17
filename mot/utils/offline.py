import logging
import numpy as np


def read_track_file(file_path):
    return np.loadtxt(file_path, delimiter=',')


def fill_gaps(tracklet, max_gap=10):
    logging.info('Gap-filling: Target #{}, length before filling: {}'.format(tracklet.id, len(tracklet.box_history)))
    box_history = tracklet.box_history
    output_trajectory = []
    current_frame = -1
    for i in range(len(box_history)):
        if current_frame > -1 and box_history[i][0] - current_frame > 1 and box_history[i][
            0] - current_frame <= max_gap:
            box_gap = box_history[i][1] - output_trajectory[-1][1]
            frame_gap = box_history[i][0] - current_frame
            unit = box_gap / frame_gap
            for j in range(current_frame + 1, box_history[i][0]):
                output_trajectory.append((j, output_trajectory[-1][1] + unit * (j - current_frame)))
        output_trajectory.append(box_history[i])
        current_frame = box_history[i][0]
    logging.info('Gap-filling: After filling: {}'.format(len(output_trajectory)))
    return output_trajectory


def remove_short_tracks(tracklets, min_time_lived=30):
    outputs = []
    removed = []
    for id, track in tracklets:
        if len(track) >= min_time_lived:
            outputs.append((id, track))
        else:
            removed.append((id, track))
    logging.info('Short-cleaning: {} tracks removed ({} lines in total)'.format(len(removed),
                                                                                sum(map(lambda t: len(t[1]), removed))))
    return outputs
