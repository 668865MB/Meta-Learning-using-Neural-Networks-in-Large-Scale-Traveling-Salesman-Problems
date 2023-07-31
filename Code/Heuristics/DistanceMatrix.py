def Distmatrix(filename):
    import numpy as np

    # Open file and read in coordinates
    with open(filename, 'r') as file:
        coordinates = [line.strip() for line in file.readlines()]
        
    processed_coordinates = []
    for line in coordinates:
        if line != 'EOF':
            values = line.split()
            processed_coordinates.append((values[1], values[2]))

    coordinates = processed_coordinates

    # Create array to hold coordinates
    n = len(coordinates)
    coords = np.zeros((n, 2))

    # Fill in coordinates array
    for i in range(n):
        line = coordinates[i]
        coords[i, 0] = float(line[0])
        coords[i, 1] = float(line[1])

    # Calculate distance matrix
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist = np.sqrt((coords[i,0] - coords[j,0])**2 + (coords[i,1] - coords[j,1])**2)
            dist_matrix[i,j] = dist
            dist_matrix[j,i] = dist

    # Output distance matrix
    return dist_matrix, coords
