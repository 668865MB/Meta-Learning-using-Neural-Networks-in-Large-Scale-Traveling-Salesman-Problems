import numpy as np
from DistanceMatrix import Distmatrix

def nearest_neighbor_BN(dist_matrix):
    # Number of cities
    n = len(dist_matrix)
    
    Ecost_quantile = np.percentile(dist_matrix[np.nonzero(dist_matrix)], 10)
    # Create a list of unvisited cities
    unvisited = list(range(1, n))

    # Start at the first city
    current_city = 0

    # Create a list to store the tour
    tour = [current_city]

    # Initialize the length of the tour to 0
    tour_length = 0

    # Repeat until all cities have been visited
    while unvisited:
        # Find the nearest neighbor of the current city
        nearest_neighbor = min(unvisited, key=lambda city: dist_matrix[current_city][city] if dist_matrix[current_city][city] >= Ecost_quantile else float('inf'))

        # Add the nearest neighbor to the tour
        tour.append(nearest_neighbor)

        # Update the length of the tour
        tour_length += dist_matrix[current_city][nearest_neighbor]

        # Move to the nearest neighbor
        current_city = nearest_neighbor

        # Remove the nearest neighbor from the list of unvisited cities
        unvisited.remove(nearest_neighbor)

    # Add the distance from the last city to the starting city
    tour_length += dist_matrix[current_city][0]
    tour.append(tour[0])
    return tour_length, tour

