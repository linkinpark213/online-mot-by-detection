import numpy as np
import scipy.cluster.hierarchy as sch


def hierarchical_cluster(M: np.ndarray, t: float, linkage_method: str = 'average',
                         criterion: str = 'maxclust'):
    """
    Perform hierarchical correlation cluster on a given DISTANCE matrix.

    Args:
        M: A 2D numpy array (N * N). The distance matrix between every point pair.
        t: A floating number indicating the max distance between any points in a cluster.
        linkage_method: A string indicating the linkage method. Default is 'average'.
        criterion: A string indicating the cluster criterion. Default is 'maxclust'.

    Returns:
        A 1D numpy array. The cluster IDs of each point.
    """
    # print('Affinity matrix')
    # for line in M:
    #     print(line)

    iu = np.triu_indices(len(M), 1, len(M))
    D = M[iu]
    Z = sch.linkage(D, method=linkage_method)
    clusterIDs = sch.fcluster(Z, t, criterion=criterion)

    return clusterIDs
