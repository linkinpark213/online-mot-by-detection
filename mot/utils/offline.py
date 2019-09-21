import logging
import numpy as np


def read_track_file(file_path):
    return np.loadtxt(file_path, delimiter=',')


def fill_gaps(tracklets, max_gap=10):
    output_trajectories = []
    for id, track in tracklets:
        logging.info('Gap-filling: Target #{}, length before filling: {}'.format(id, len(track)))
        output_trajectory = []
        current_frame = -1
        for i in range(len(track)):
            if current_frame > -1 and track[i][0] - current_frame > 1 and track[i][0] - current_frame <= max_gap:
                box_gap = track[i][1] - output_trajectory[-1][1]
                frame_gap = track[i][0] - current_frame
                unit = box_gap / frame_gap
                for j in range(current_frame + 1, track[i][0]):
                    output_trajectory.append((j, output_trajectory[-1][1] + unit * (j - current_frame)))
            current_frame = track[i][0]
            output_trajectory.append(track[i])
        logging.info('Gap-filling: After filling: {}'.format(len(output_trajectory)))
        output_trajectories.append(output_trajectory)
    return output_trajectories


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
