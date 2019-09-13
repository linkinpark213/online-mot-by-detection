import os
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

    n_deleted_tracks = 0
    n_deleted_records = 0
    for id, track in tracks.items():
        if track.__len__() < min_time_lived:
            n_deleted_tracks += 1
            n_deleted_records += len(tracks[int(id)])
            tracks[int(id)] = np.array([])
    print('{} track(s) ({} records) deleted'.format(n_deleted_tracks, n_deleted_records))

    collection = []
    for id, track in tracks.items():
        for line in track:
            collection.append(line)

    return np.array(collection)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str, help='Path to the input result file or directory of input result files')
    parser.add_argument('--min_time_lived', type=int, default=30, help='Minimum time lived of tracks')
    parser.add_argument('--output_path', type=str, default='cleaned', help='Path to the output file\'s directory')
    args = parser.parse_args()

    input_files = []
    if os.path.isfile(args.input_path):
        input_files = [args.input_path]
    elif os.path.isdir(args.input_path):
        input_files = [os.path.join(args.input_path, filename) for filename in os.listdir(args.input_path)]

    if not os.path.isdir(args.output_path):
        os.makedirs(args.output_path)

    for file_path in input_files:
        print('Processing {}'.format(file_path))
        input_filename = file_path.split('/')[-1]
        track_results = read_track_file(file_path)
        track_results = remove_short_tracks(track_results, args.min_time_lived)

        output_filename = os.path.join(args.output_path, input_filename)
        output_file = open(output_filename, 'w+')

        for line in track_results:
            output_file.write(
                '{:d}, {:d}, {:.2f}, {:.2f}, {:.2f}, {:.2f}\n'.format(int(line[0]), int(line[1]), line[2], line[3],
                                                                      line[4], line[5]))

        output_file.close()