import numpy as np
import networkx as nx
from sklearn.cluster import DBSCAN
from sklearn.metrics import calinski_harabasz_score
from Mst import mst


def features(distance_matrix,coords):

    #vertices info
    Vnum = len(distance_matrix)
    

    Vcost = []
    NN = []
    for i in distance_matrix:
        Vcost.append(np.sum(i)/(Vnum-1))
        NN.append( np.amin(i[i > 0]))

    Vcost_min = min(Vcost)  
    Vcost_max = max(Vcost) 
    Vcost_average = np.mean(Vcost)
    Vcost_std = np.std(Vcost)
    Vcost_median = np.median(Vcost)
    Vcost_NN = sum(NN)

    #edges
    Enum = Vnum*(Vnum-1)/2    
    Ecost_min = np.amin(distance_matrix[np.nonzero(distance_matrix)])
    Ecost_max = np.max(distance_matrix)
    Ecost_average = np.mean(distance_matrix[np.nonzero(distance_matrix)])
    Ecost_sd = np.std(distance_matrix[np.nonzero(distance_matrix)])
    Ecost_median = np.median(distance_matrix[np.nonzero(distance_matrix)])

    #eigenvalues of distance matrix
    eigenvalues = np.linalg.eigvals(distance_matrix)
    EIG_sd = np.std(eigenvalues)
    EIG_ratio = np.max(eigenvalues)/np.min(eigenvalues)

    # Clustering coefficient
    adj_matrix = np.where(distance_matrix > 0, 1, 0)
    n_nodes = adj_matrix.shape[0]
    weights = np.multiply(adj_matrix, distance_matrix)
    local_cc = np.zeros(n_nodes)
    global_cc = 0

    for i in range(n_nodes):
        # Get the neighbors of node i
        neighbors = np.nonzero(adj_matrix[i])[0]

        # Calculate the sum of the weights of the edges that exist between the neighbors of node i
        w_i = np.sum(weights[neighbors][:, neighbors]) / 2

        # Calculate the local clustering coefficient of node i
        k_i = len(neighbors)
        if k_i > 1:
            local_cc[i] = (2 * w_i) / (k_i * (k_i - 1))

    # Calculate the global clustering coefficient of the network
    if n_nodes > 0:
        global_cc = np.mean(local_cc)

    import matplotlib.pyplot as plt

    # coords is a list of tuples containing the x and y coordinates
    #x_coords = [coord[0] for coord in coords]
    #y_coords = [coord[1] for coord in coords]

    #plt.scatter(x_coords, y_coords)
    #plt.show()

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


    # Print the best clustering and its score
    #print("Best clustering:")
    #print(best_clustering.labels_)
    #print("Calinski-Harabasz score:", best_score)
    #print("num_clusters:", num_clusters)
    #print("num_outliers:", num_outliers)
    #print("eps,min_samples:", best_params)

        # Plot the data with different colors for each cluster
    #fig, ax = plt.subplots(figsize=(8, 6))
    #unique_labels = set(best_clustering.labels_)
    #colors = [plt.cm.Spectral(each)
    #        for each in np.linspace(0, 1, len(unique_labels))]
    #for k, col in zip(unique_labels, colors):
    #    if k == -1:
     #       # Black used for noise.
      #      col = [0, 0, 0, 1]

       # class_member_mask = (best_clustering.labels_ == k)

        #xy = coords[class_member_mask]
        #ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
         #       markeredgecolor='k', markersize=5)

    #ax.set_title('DBSCAN Clustering')
    #plt.show()

    MST = mst(distance_matrix)
    dm_sorted = np.sort(distance_matrix, axis=None)
    Edge_ord  = np.sum(dm_sorted[dm_sorted != 0][:Vnum])

    return Vnum,Vcost_min,Vcost_max ,Vcost_median,Vcost_average,Vcost_std, Enum, Ecost_min,Ecost_max,Ecost_median,Ecost_average,Ecost_sd,EIG_ratio, EIG_sd,global_cc, num_outliers, num_clusters, MST, Edge_ord
            
