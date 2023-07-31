import numpy as np

def nearest_insertion(dist_matrix):
    n = dist_matrix.shape[0]
    unvisited = set(range(n))
    start = 0
    unvisited.remove(start)
    tour = [start]
    current = start

    while unvisited:
        #nearest = min(unvisited, key=lambda x: min(dist_matrix[x][city] for city in tour))
        nearest = min(unvisited, key=lambda x: dist_matrix[current][x])
        min_insert_cost = float('inf')
        for i in range(len(tour)):
            cost = dist_matrix[tour[i]][nearest] + dist_matrix[nearest][tour[(i+1)%len(tour)]] - dist_matrix[tour[i]][tour[(i+1)%len(tour)]]
            if cost < min_insert_cost:
                min_insert_cost = cost
                insert_index = i
        tour.insert(insert_index+1, nearest)
        unvisited.remove(nearest)
        current = nearest
    tour.append(tour[0])
    tour_length = sum(dist_matrix[tour[i]][tour[(i+1)%n]] for i in range(n))
    return tour_length, tour