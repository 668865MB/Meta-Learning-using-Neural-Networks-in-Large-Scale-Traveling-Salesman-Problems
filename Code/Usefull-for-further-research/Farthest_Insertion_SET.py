import numpy as np

def farthest_insertion_SET(dist_matrix, cluster):

    n = dist_matrix.shape[0]
    rows, cols = dist_matrix.shape

    for i in range(rows):
        for j in range(cols):
            if cluster[i] == cluster[j] and  cluster[i] != -1 :
                dist_matrix[i][j] = 0

    unvisited = set(range(n))
    start = 0
    tour = [start]
    current = start

    current_cities = [current]
    if cluster[current] != -1:
        for i in range(len(cluster)):
            if cluster[current] == cluster[i]:
                unvisited.remove(i)
                current_cities.append(i)
    else:
        unvisited.remove(current)



    def find_farthest_unvisited(current_cities):
        farthest_distance = float(-1)

        for i in range(len(current_cities)):        
            for neighbor in unvisited:

                if dist_matrix[current_cities[i]][neighbor] > farthest_distance and current_cities[i] != neighbor:
                    From = current_cities[i]
                    To = neighbor
                    farthest = dist_matrix[current_cities[i]][neighbor]
        edge = From,To
        return edge

    while unvisited:
        #nearest = min(unvisited, key=lambda x: min(dist_matrix[x][city] for city in tour))
        nearest = find_farthest_unvisited(current_cities)
        min_insert_cost = float('inf')
        for i in range(len(tour)):
            cost = dist_matrix[tour[i]][nearest[1]] + dist_matrix[nearest[1]][tour[(i+1)%len(tour)]] - dist_matrix[tour[i]][tour[(i+1)%len(tour)]]
            if cost < min_insert_cost:
                min_insert_cost = cost
                insert_index = i
        tour.insert(insert_index+1, nearest[1])
        #tour.insert(insert_index, nearest[0])

        current_cities = [nearest[1]]
        if cluster[nearest[1]] != -1:
            for i in range(len(cluster)):
                if cluster[nearest[1]] == cluster[i]:
                    unvisited.remove(i)
                    current_cities.append(i)
        else:
            unvisited.remove(nearest[1])

    tour.append(tour[0])
    edges = []
    tour_length = 0
    i = 0
    for i in range(len(tour)-1):
        edges.append((tour[i],tour[i+1]))
    
    for i in range(len(edges)):
        edge = edges[i]
        if cluster[edge[0]] != -1:
            best_start = edge[0]
            cost = dist_matrix[edge[0]][edge[1]]
            for j in range(len(cluster)):
                if cluster[edge[0]] == cluster[j] and dist_matrix[j][edge[1]] < cost:
                    best_start = j
                    cost = dist_matrix[j][edge[1]]
            edges[i] = (best_start, edge[1])
        tour_length += dist_matrix[edges[i][0]][edges[i][1]]



    return tour_length, edges
        
