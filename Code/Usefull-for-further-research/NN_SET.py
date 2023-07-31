import numpy as np
from DistanceMatrix import Distmatrix

def nearest_neighbor_SET(dist_matrix,cluster):



    n = dist_matrix.shape[0]

    # Create a list of unvisited cities
    unvisited = list(range(0, n))


    # Start at the first city
    current_city = 0

    # Create a list to store the tour
    tour = []

    # Initialize the length of the tour to 0
    tour_length = 0

    current_cities = [current_city]
    if cluster[current_city] != -1:
        for i in range(len(cluster)):
            if cluster[current_city] == cluster[i]:
                unvisited.remove(i)
                current_cities.append(i)
    else:
        unvisited.remove(current_city)
    

    

    # Function to find the nearest unvisited neighbor from a given city
    def find_nearest_unvisited(current_cities):
        nearest_distance = float('inf')

        for i in range(len(current_cities)):        
            for neighbor in unvisited:

                if dist_matrix[current_cities[i]][neighbor] < nearest_distance and current_cities[i] != neighbor:
                    From = current_cities[i]
                    To = neighbor
                    nearest_distance = dist_matrix[current_cities[i]][neighbor]
        edge = (From,To)
        return edge



    # Repeat until all cities have been visited
    while unvisited:
        # Find the nearest neighbor of the current city

        nearest_neighbor = find_nearest_unvisited(current_cities)
  
        # Add the nearest neighbor to the tour
        tour.append(nearest_neighbor)

        tour_length += dist_matrix[nearest_neighbor[0]][nearest_neighbor[1]]

        # Move to the nearest neighbor
        current_cities = [nearest_neighbor[1]]


        if cluster[nearest_neighbor[1]] != -1:
            for i in range(len(cluster)):
                if cluster[nearest_neighbor[1]] == cluster[i]:
                    unvisited.remove(i)
                    current_cities.append(i)
        else:
            unvisited.remove(nearest_neighbor[1])

    
    Begin_cities = [tour[0][0]]
    if cluster[tour[0][0]] != -1:
        for i in range(len(cluster)):
            if cluster[tour[0][0]] == cluster[i]:
                Begin_cities.append(i)

    nearest_distance = float('inf')
    for i in Begin_cities:
        for j in current_cities:
             if dist_matrix[i][j] < nearest_distance:
                    end_edge = i
                    begin_edge = j
                    nearest_distance = dist_matrix[i][j]
    #if cluster[tour[-1][1]] != -1 or cluster[tour[0][0]] != -1:
    #tour.append((tour[0][0],tour[-1][1]))
    
    tour.append((end_edge,begin_edge))
    tour_length += dist_matrix[end_edge][begin_edge]




 
    return tour_length, tour

