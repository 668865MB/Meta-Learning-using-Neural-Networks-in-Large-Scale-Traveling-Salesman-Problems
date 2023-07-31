import numpy as np
import networkx as nx
from sklearn.cluster import DBSCAN
from sklearn.metrics import calinski_harabasz_score



def dbscan(distance_matrix,coords):

    n = distance_matrix.shape[0]
    Ecost_min = np.amin(distance_matrix[np.nonzero(distance_matrix)])

    # Define the parameter space to search over
    eps_range = [Ecost_min*3,Ecost_min*4,Ecost_min*5,Ecost_min*6,Ecost_min*7,Ecost_min*8,Ecost_min*9,Ecost_min*10]
    min_samples_range = [3,4,5]

    # Initialize variables to keep track of the best clustering and its score
    best_clustering = None
    best_score = -1
    num_clusters = None
    num_outliers = None



    # Search over the parameter space
    for eps in eps_range:
        for min_samples in min_samples_range:
            
            # Cluster the data
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
            
            try:
                # Evaluate the clustering using the Calinski-Harabasz index
                score = calinski_harabasz_score(coords, clustering.labels_)
            except ValueError:
                continue
            
            
            # Keep track of the best clustering and its score
            if score > best_score:
                best_clustering = clustering
                best_score = score
                best_params = (eps, min_samples)
                num_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
                num_outliers = np.sum(clustering.labels_ == -1)

    if best_clustering == None:
        r = [-1]*n
    else:
        r = best_clustering.labels_     

    return r    