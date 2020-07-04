import numpy as np
import scipy.spatial.distance as ssd
import scipy.cluster.hierarchy as sch


def hierarchical_cluster(features: np.ndarray, t: float, distance_metric: str = 'cosine',
                         linkage_method: str = 'average', criterion: str = 'maxclust'):
    """
    Perform hierarchical correlation cluster on a given DISTANCE matrix.

    Args:
        features: A 2D numpy array (N * P). The N feature vectors with P dimensions.
        t: A floating number indicating the max distance between any points in a cluster, or max number of clusters. (Depending on the criterion.)
        distance_metric: A string indicating the distance metric. Default is 'cosine'.
        linkage_method: A string indicating the linkage method. Default is 'average'.
        criterion: A string indicating the cluster criterion. Default is 'maxclust'.

    Returns:
        A 1D numpy array. The cluster IDs of each point.
    """
    if not np.any(features != 0):
        raise AssertionError('Feature vector can\'t be all zero')

    D = ssd.pdist(features, metric=distance_metric)
    D = np.clip(D, 0, 1)
    Z = sch.linkage(D, method=linkage_method)
    clusterIDs = sch.fcluster(Z, t, criterion=criterion)

    return clusterIDs
