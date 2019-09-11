import argparse

import numpy as np


def read_track_file(file_path):
    return np.loadtxt(file_path, delimiter=',')


def remove_short_tracks(track_results, min_time_lived):
    tracks = {}
    for line in track_results:
        id = line[1]
        if id in tracks.keys():
            tracks[id].append(line)
        else:
            tracks[id] = [line]

    for id, track in tracks.items():
        if track.__len__() < min_time_lived:
            track[id] = np.array([])

    collection = []
    for id, track in tracks.items():
        collection.append(track)

    return np.array(collection)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str, help='Path to the input result file')
    parser.add_argument('--min_time_lived', type=int, default=10, help='Minimum time lived of tracks')
    args = parser.parse_args()

    track_results = read_track_file(args.input_file)
    track_results = remove_short_tracks(track_results, args.min_time_lived)

    filename = args.input_file.split('/')[-1]
    output_file_name = filename + '_out.txt'
    output_file = open(output_file_name, 'w+')


    for line in track_results:
        output_file.write(
            '{:d}, {:d}, {:.2f}, {:.2f}, {:.2f}, {:.2f}\n'.format(line[0], line[1], line[2], line[3], line[4], line[5]))

    output_file.close()
