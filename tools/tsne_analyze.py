import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import mot.utils.visualize as vis

if __name__ == '__main__':
    tsne = TSNE(n_components=2)

    print('Reading features file')
    data = np.loadtxt('features.txt')
    print('Data shape: ', data.shape)

    corresponding_ids = data[:, 0]
    frame_nums = data[:, 1]
    all_features = data[:, 2:]

    ranges = [(1, 100), (1, 200), (1, 500)]

    for frame_range in ranges:
        available_indices = np.where(frame_nums >= frame_range[0] and frame_nums <= frame_range[1])
        period_features = all_features[available_indices]
        period_ids = corresponding_ids[available_indices]

        features_embedded = tsne.fit_transform(period_features)

        plt.figure()
        for i, point in enumerate(features_embedded):
            plt.scatter(point[0], point[1],
                        c=np.array([list(vis._colors[int(period_ids[i]) % len(vis._colors)])]) / 255.0)
        plt.show()