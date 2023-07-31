import numpy as np

def nearest_insertion_bn(dist_matrix):
    n = dist_matrix.shape[0]
    Ecost_quantile = np.percentile(dist_matrix[np.nonzero(dist_matrix)], 10)
    unvisited = set(range(n))
    start = 0
    unvisited.remove(start)
    tour = [start]
    current = start
    insert_index = -1
    shortest_edge = float('inf')

    while unvisited:
        #nearest = min(unvisited, key=lambda x: min(dist_matrix[x][city] for city in tour))
        nearest = min(unvisited, key=lambda x: dist_matrix[current][x])
        min_insert_cost = float('inf')
        for i in range(len(tour)):
            cost = dist_matrix[tour[i]][nearest] + dist_matrix[nearest][tour[(i+1)%len(tour)]] - dist_matrix[tour[i]][tour[(i+1)%len(tour)]]
            if Ecost_quantile < cost < min_insert_cost:
                min_insert_cost = cost
                insert_index = i
        tour.insert(insert_index+1, nearest)
        unvisited.remove(nearest)
        current = nearest
    tour.append(tour[0])
    
    tour_length = 0
    shortest_edge = float('inf')
    for i in range(n):
        contribution = dist_matrix[tour[i]][tour[(i+1)]  ]                            
        tour_length += contribution

        # Update the shortest edge if necessary
        if contribution < shortest_edge:
            shortest_edge = contribution

    print(shortest_edge)
    return tour_length, tour