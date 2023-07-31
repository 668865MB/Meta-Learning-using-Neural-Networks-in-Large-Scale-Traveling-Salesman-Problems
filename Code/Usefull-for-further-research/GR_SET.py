import numpy as np
import sys


def sorted_edges_algorithm_SET(distance_matrix,cluster):
    sys.setrecursionlimit(5000000)
    n = distance_matrix.shape[0]

    num_clusters = len(set(cluster)) - (1 if -1 in cluster else 0)
    num_outliers = np.sum(cluster == -1)



    # Create a list of unvisited cities
    unvisited = list(range(0, n))
    edges = [(i, j, distance_matrix[i][j]) for i in range(n) for j in range(i+1, n)]
    shadowtour = []
    edges_to_remove = []
    

    edges_to_keep = []

    for i in edges:
        if cluster[i[0]] != -1 and cluster[i[0]] == cluster[i[1]]:
            shadowtour.append(i)
        else:
            edges_to_keep.append(i)

    edges = edges_to_keep

    #for i in edges:

     #   if cluster[i[0]] != -1:
      #      if cluster[i[0]] == cluster[i[1]]:
          
       #         shadowtour.append(i)
    #for edge in shadowtour:
     #   edges.remove(edge)

    edges.sort(key=lambda x: x[2])
    cluster_degree = [0] * num_clusters 
    outlier_degree = [0] * n 

    
    tour = []
       

    for i in edges: 

        #CASE 1 BOTH OUTLIERS
        if cluster[i[0]] == -1 == cluster[i[1]]: 

            if outlier_degree[i[0]] <= 1 and outlier_degree[i[1]] <= 1:
                if outlier_degree[i[0]] == 1 and outlier_degree[i[1]] == 1 and len(tour) == num_clusters+num_outliers-1:
                    tour.append(i)

                    shadowtour.append(i)
                    outlier_degree[i[0]] +=1
                    outlier_degree[i[1]] +=1

                elif outlier_degree[i[0]] < 1 or outlier_degree[i[1]] < 1:
                    tour.append(i)

                    shadowtour.append(i)
                    outlier_degree[i[0]] +=1
                    outlier_degree[i[1]] +=1
                
                elif outlier_degree[i[0]] == 1 and outlier_degree[i[1]] == 1 and has_path(shadowtour,i[0],i[1]) == False:
                    tour.append(i)

                    shadowtour.append(i)
                    outlier_degree[i[0]] +=1
                    outlier_degree[i[1]] +=1

        #CASE 2 CLUSTER TO OUTLIER
        if cluster[i[0]] == -1 and cluster[i[1]] != -1: 

            if outlier_degree[i[0]] <= 1 and cluster_degree[cluster[i[1]]] <= 1:
                if outlier_degree[i[0]] == 1 and cluster_degree[cluster[i[1]]] == 1 and len(tour) == num_clusters+num_outliers-1:
                    tour.append(i)
                    

                    outlier_degree[i[0]] +=1
                    cluster_degree[cluster[i[1]]] +=1
                    for k in edges:
                        if k[0] == i[0] and cluster[i[1]] == cluster[k[1]]:
                            shadowtour.append((i[0],k[1]))



                elif outlier_degree[i[0]] < 1 or cluster_degree[cluster[i[1]]] < 1:
                    tour.append(i)

                    outlier_degree[i[0]] +=1
                    cluster_degree[cluster[i[1]]] +=1
                    for k in edges:
                        if k[0] == i[0] and cluster[i[1]] == cluster[k[1]]:
                            shadowtour.append((i[0],k[1]))
                
                elif outlier_degree[i[0]] == 1 and cluster_degree[cluster[i[1]]] == 1 and has_path(shadowtour,i[0],i[1]) == False:
                    tour.append(i)

                    outlier_degree[i[0]] +=1
                    cluster_degree[cluster[i[1]]] +=1
                    for k in edges:
                        if k[0] == i[0] and cluster[i[1]] == cluster[k[1]]:
                            shadowtour.append((i[0],k[1]))

        #CASE 3 OUTLIER TO CLUSTER    
        if cluster[i[0]] != -1 and cluster[i[1]] == -1: 

            if outlier_degree[i[1]] <= 1 and cluster_degree[cluster[i[0]]] <= 1:
                if outlier_degree[i[1]] == 1 and cluster_degree[cluster[i[0]]] == 1 and len(tour) == num_clusters+num_outliers-1:
                    

                    tour.append(i)
                    outlier_degree[i[1]] +=1
                    cluster_degree[cluster[i[0]]] +=1
                    for k in edges:
                        if k[1] == i[1] and cluster[i[0]] == cluster[k[0]]:
                            shadowtour.append((i[1],k[0]))



                elif outlier_degree[i[1]] < 1 or cluster_degree[cluster[i[0]]] < 1:


                    tour.append(i)
                    outlier_degree[i[1]] +=1
                    cluster_degree[cluster[i[0]]] +=1
                    for k in edges:
                        if k[1] == i[1] and cluster[i[0]] == cluster[k[0]]:
                            shadowtour.append((i[1],k[0]))
                
                elif outlier_degree[i[1]] == 1 and cluster_degree[cluster[i[0]]] == 1 and has_path(shadowtour,i[0],i[1]) == False:
                   
                   

                   
                    tour.append(i)
                    outlier_degree[i[1]] +=1
                    cluster_degree[cluster[i[0]]] +=1
                    for k in edges:
                        if k[1] == i[1] and cluster[i[0]] == cluster[k[0]]:
                            shadowtour.append((i[1],k[0]))
                


        #CASE 4 CLUSTER TO CLUSTER    
        if cluster[i[0]] != -1 and cluster[i[1]] != -1: 
            if cluster_degree[cluster[i[1]]] <= 1 and cluster_degree[cluster[i[0]]] <= 1:
                    if cluster_degree[cluster[i[1]]] == 1 and cluster_degree[cluster[i[0]]] == 1 and len(tour) == num_clusters+num_outliers-1:
                        

                        
                        tour.append(i)
                        cluster_degree[cluster[i[1]]] +=1
                        cluster_degree[cluster[i[0]]] +=1
                        for k in edges:
                            if cluster[k[1]] == cluster[i[1]] and cluster[i[0]] == cluster[k[0]]:
                                shadowtour.append((i[1],k[0]))



                    elif cluster_degree[cluster[i[1]]] < 1 or cluster_degree[cluster[i[0]]] < 1:
                        

                        tour.append(i)
                        cluster_degree[cluster[i[1]]] +=1
                        cluster_degree[cluster[i[0]]] +=1
                        for k in edges:
                            if cluster[k[1]] == cluster[i[1]] and cluster[i[0]] == cluster[k[0]]:
                                shadowtour.append((i[1],k[0]))
                    
                    elif cluster_degree[cluster[i[1]]] == 1 and cluster_degree[cluster[i[0]]] == 1 and has_path(shadowtour,i[0],i[1]) == False:
                        tour.append(i)

                        
                        cluster_degree[cluster[i[1]]] +=1
                        cluster_degree[cluster[i[0]]] +=1
                        for k in edges:
                            if cluster[k[1]] == cluster[i[1]] and cluster[i[0]] == cluster[k[0]]:
                                shadowtour.append((i[1],k[0]))





    T = [tour[0][0],tour[0][1]]
    C = 0
 

    for i in tour:
        C += i[2]

    


    #for i in range(1,len(tour)):
       # if tour[i][0] == T[0]:
          #  T.append(tour[i][1])
           # C += tour[i][2]
        #elif tour[i][1] == T[0]:
         #   T.append(tour[i][0])
          #  C += tour[i][2]
        
    
    return C, tour

def has_path(edges, x, y):
    # Create a dictionary to store the adjacency list of the graph
    graph = {}
    for e in edges:
        if e[0] not in graph:
            graph[e[0]] = []
        graph[e[0]].append(e[1])
        if e[1] not in graph:
            graph[e[1]] = []
        graph[e[1]].append(e[0])
    
    # Create a set to keep track of visited nodes
    visited = set()
    
    def dfs(node):
        visited.add(node)
        if node == y:
            return True
        for neigh in graph[node]:
            if neigh not in visited:
                if dfs(neigh):
                    return True
        return False
    
    # Check for a path starting from node x
    return dfs(x)