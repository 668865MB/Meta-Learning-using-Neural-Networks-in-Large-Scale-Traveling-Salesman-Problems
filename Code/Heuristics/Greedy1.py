import numpy as np
import sys


def sorted_edges_algorithm(distance_matrix):
    sys.setrecursionlimit(5000000)
    n = distance_matrix.shape[0]
    edges = [(i, j, distance_matrix[i][j]) for i in range(n) for j in range(i+1, n)]
    edges.sort(key=lambda x: x[2])
    degree = [0] * n
    tour = []
    

    for i in edges: 
        if degree[i[0]] <= 1 and degree[i[1]] <= 1:
            if degree[i[0]] == 1 and degree[i[1]] == 1 and len(tour) == n-1:
                tour.append(i)
                degree[i[0]] +=1
                degree[i[1]] +=1

            elif degree[i[0]] < 1 or degree[i[1]] < 1:
                tour.append(i)
                degree[i[0]] +=1
                degree[i[1]] +=1
            
            elif degree[i[0]] == 1 and degree[i[1]] == 1 and has_path(tour,i[0],i[1]) == False:
                tour.append(i)
                degree[i[0]] +=1
                degree[i[1]] +=1



    T = [tour[0][0],tour[0][1]]
    C = tour[0][2]
    Used = [False]*len(tour)
    Used[0] = True

    while len(T) < n:
        for i in range(1,(len(tour))):
            if tour[i][0] == T[-1] and Used[i] == False :
                T.append(tour[i][1])
                Used[i] = True
                C += tour[i][2]
            elif tour[i][1] == T[-1] and Used[i] == False:
                T.append(tour[i][0])
                C += tour[i][2]
                Used[i] = True


    #for i in range(1,len(tour)):
       # if tour[i][0] == T[0]:
          #  T.append(tour[i][1])
           # C += tour[i][2]
        #elif tour[i][1] == T[0]:
         #   T.append(tour[i][0])
          #  C += tour[i][2]
        
    
    return C, T

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