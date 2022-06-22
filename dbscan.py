# -*- coding: utf-8 -*-

# A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise
# Martin Ester, Hans-Peter Kriegel, Jörg Sander, Xiaowei Xu
# dbscan: density based spatial clustering of applications with noise

import numpy as np
import math
from scipy.spatial import distance
from sklearn.cluster import cluster_optics_dbscan
UNCLASSIFIED = False
NOISE = None

dists = []
# def _dist(p,q):
# 	return math.sqrt(np.power(p-q,2).sum())
# def _distances(m):
#     d=distance.cdist(m,m, 'euclidean')
#     return d

	#return math.sqrt(np.power(p-q,2).sum())
def _eps_neighborhood(p,q,eps):
	return dists[p][q] < eps

def _region_query(m, point_id, eps):
    n_points = m.shape[1]
    m2=m.transpose()
    seeds = []
    for i in range(0, n_points):
        if _eps_neighborhood(np.where(m2==m[:,point_id])[0][0], np.where(m2==m[:,i])[0][0], eps):
            seeds.append(i)
    return seeds# -*- coding: utf-8 -*-

# A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise
# Martin Ester, Hans-Peter Kriegel, Jörg Sander, Xiaowei Xu
# dbscan: density based spatial clustering of applications with noise

import numpy as np
import math
from scipy.spatial import distance
from sklearn.cluster import cluster_optics_dbscan
UNCLASSIFIED = False
NOISE = None

dists = []
# def _dist(p,q):
# 	return math.sqrt(np.power(p-q,2).sum())
# def _distances(m):
#     d=distance.cdist(m,m, 'euclidean')
#     return d

	#return math.sqrt(np.power(p-q,2).sum())
def _eps_neighborhood(p,q,eps):
    print("Run _eps_neighborhood")
    print("End Run _eps_neighborhood")
    return dists[p][q] < eps

def _region_query(m, point_id, eps):
    print("Run _region_query")
    n_points = m.shape[1]
    m2=m.transpose()
    seeds = []
    for i in range(0, n_points):
        if _eps_neighborhood(np.where(m2==m[:,point_id])[0][0], np.where(m2==m[:,i])[0][0], eps):
            seeds.append(i)
    print("End Run  _region_query")
    return seeds

def _expand_cluster(m, classifications, point_id, cluster_id, eps, min_points):
    print("Run _expand_cluster")
    seeds = _region_query(m, point_id, eps)
    if len(seeds) < min_points:
        classifications[point_id] = NOISE
        return False
    else:
        classifications[point_id] = cluster_id
        for seed_id in seeds:
            classifications[seed_id] = cluster_id
            
        while len(seeds) > 0:
            current_point = seeds[0]
            results = _region_query(m, current_point, eps)
            if len(results) >= min_points:
                for i in range(0, len(results)):
                    result_point = results[i]
                    if classifications[result_point] == UNCLASSIFIED or \
                       classifications[result_point] == NOISE:
                        if classifications[result_point] == UNCLASSIFIED:
                            seeds.append(result_point)
                        classifications[result_point] = cluster_id
            seeds = seeds[1:]
        print("End Run _expand_cluster")
        return True
        
def dbscan(m, eps, min_points):
    print("Run dbscan")

    """Implementation of Density Based Spatial Clustering of Applications with Noise
    See https://en.wikipedia.org/wiki/DBSCAN
    
    scikit-learn probably has a better implementation
    
    Uses Euclidean Distance as the measure
    
    Inputs:
    m - A matrix whose columns are feature vectors
    eps - Maximum distance two points can be to be regionally related
    min_points - The minimum number of points to make a cluster
    
    Outputs:
    An array with either a cluster id number or dbscan.NOISE (None) for each
    column vector in m.
    """
    cluster_id = 1
    n_points = m.shape[1]
    print(n_points)
    classifications = [UNCLASSIFIED] * n_points
    for point_id in range(0, n_points):
        point = m[:,point_id]
        if classifications[point_id] == UNCLASSIFIED:
            if _expand_cluster(m, classifications, point_id, cluster_id, eps, min_points):
                cluster_id = cluster_id + 1
    print("End Run dbscan")
    return classifications

def main(d,eps,min_points):
    print("Run main")
    global dists
    print("Run distance")
    dists = distance.cdist(d.transpose(),d.transpose(), 'euclidean')
    print("end Run distance")
    clusters=dbscan(d, eps, min_points)
    print("end Run main")
    return clusters
def test_dbscan():
    m = np.matrix('1 1.2 0.8 3.7 3.9 3.6 10; 1.1 0.8 1 4 3.9 4.1 10')
    eps = 0.5
    min_points = 2
    assert dbscan(m, eps, min_points) == [1, 1, 1, 2, 2, 2, None]


def _expand_cluster(m, classifications, point_id, cluster_id, eps, min_points):
    seeds = _region_query(m, point_id, eps)
    if len(seeds) < min_points:
        classifications[point_id] = NOISE
        return False
    else:
        classifications[point_id] = cluster_id
        for seed_id in seeds:
            classifications[seed_id] = cluster_id
            
        while len(seeds) > 0:
            current_point = seeds[0]
            results = _region_query(m, current_point, eps)
            if len(results) >= min_points:
                for i in range(0, len(results)):
                    result_point = results[i]
                    if classifications[result_point] == UNCLASSIFIED or \
                       classifications[result_point] == NOISE:
                        if classifications[result_point] == UNCLASSIFIED:
                            seeds.append(result_point)
                        classifications[result_point] = cluster_id
            seeds = seeds[1:]
        return True
        
def dbscan(m, eps, min_points):
    """Implementation of Density Based Spatial Clustering of Applications with Noise
    See https://en.wikipedia.org/wiki/DBSCAN
    
    scikit-learn probably has a better implementation
    
    Uses Euclidean Distance as the measure
    
    Inputs:
    m - A matrix whose columns are feature vectors
    eps - Maximum distance two points can be to be regionally related
    min_points - The minimum number of points to make a cluster
    
    Outputs:
    An array with either a cluster id number or dbscan.NOISE (None) for each
    column vector in m.
    """
    cluster_id = 1
    n_points = m.shape[1]
    print(n_points)
    classifications = [UNCLASSIFIED] * n_points
    for point_id in range(0, n_points):
        point = m[:,point_id]
        if classifications[point_id] == UNCLASSIFIED:
            if _expand_cluster(m, classifications, point_id, cluster_id, eps, min_points):
                cluster_id = cluster_id + 1
    return classifications

def main(d,eps,min_points):
    global dists
    dists = distance.cdist(d.transpose(),d.transpose(), 'euclidean')
    clusters=dbscan(d, eps, min_points)
    return clusters
def test_dbscan():
    m = np.matrix('1 1.2 0.8 3.7 3.9 3.6 10; 1.1 0.8 1 4 3.9 4.1 10')
    eps = 0.5
    min_points = 2
    assert dbscan(m, eps, min_points) == [1, 1, 1, 2, 2, 2, None]
