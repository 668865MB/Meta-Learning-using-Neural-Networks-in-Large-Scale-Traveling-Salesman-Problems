
def generate_sample():
    import numpy as np
    import random

    n = random.randint(100,3000)
    lambdas = ["Sparse", "Medium", "Dense"]
    lambdas = np.array(lambdas)
    i = np.random.choice(lambdas.shape[0], size=1, replace=False)
    Lambda = lambdas[i]


    #SET A
    if Lambda == "Sparse":
        scale = 100
    elif Lambda == "Medium":
        scale = 10
    elif Lambda == "Dense":
        scale = 1

    # Mean vector and covariance matrix
    mu = [0, 0]
    cov = [[10**scale, 0], [0, 10**scale]]

    def euclidean_dist(x, y):
        return np.sqrt(np.sum((x - y)**2))

    # Generate 1000 samples from a bivariate normal distribution
    samples = []

    while True:
        sample = np.random.multivariate_normal(mu, cov)
        #if all(euclidean_dist(sample, s) >= 0.00001 for s in samples):
        samples.append(sample)
        if len(samples) >= n:
            break

    selected_coords_A = np.array(samples)




    #Set B


    min_step = int(np.ceil(0.75*np.sqrt(n)))
    steps = [min_step, int(np.ceil(min_step/2)), int(np.ceil(min_step/4))]

    if Lambda == "Sparse":
        step = steps[2]
    elif Lambda == "medium":
        step = steps[1]
    elif Lambda == "Dense":
        step = steps[0]

    step = steps[2]
    grid = []
    for x in range(0, n, step):
        for y in range(0, n, step):
            grid.append((x, y))
    midpoint = (n//2, n//2)
    grid = [(x-midpoint[0], y-midpoint[1]) for x,y in grid]

    selected_coords_idx = np.random.choice(len(grid), size=n, replace=False)
    selected_coords = [grid[i] for i in selected_coords_idx]
    selected_coords_B = np.array(selected_coords)


    #Set C
    filename = r"C:\Users\Matth\OneDrive\Documenten\University\Master Thesis\Literature\usadata.txt"

    with open(filename, 'r') as file:
        coordinates = [line.strip() for line in file.readlines()]
            
        processed_coordinates = []
        for line in coordinates:
            if line != 'EOF' and len(line)>0: 
                values = line.split()
                processed_coordinates.append((values[1], values[2]))

        coordinates = processed_coordinates

        # Create array to hold coordinates
        cities = len(coordinates)
        coords = np.zeros((cities, 2))

        # Fill in coordinates array
        for i in range(cities):
            line = coordinates[i]
            coords[i, 0] = float(line[0])
            coords[i, 1] = float(line[1])


    center = np.mean(coords, axis=0)
    # Shift coordinates such that center is at origin
    coords = coords - center
    # Randomly select n coordinates
    # Replace 5 with the number of coordinates you want to select

    selected_rows = np.random.choice(coords.shape[0], size=n, replace=False)
    selected_coords = coords[selected_rows, :]


    selected_coords_C = np.array(selected_coords)


    #meanA = np.mean(selected_coords_A, axis=0)
    #stdA = np.std(selected_coords_A, axis=0)
    #selected_coords_A = (selected_coords_A - meanA) / stdA

    #meanB = np.mean(selected_coords_B, axis=0)
    #stdB = np.std(selected_coords_B, axis=0)
    #selected_coords_B = (selected_coords_B - meanB) / stdB


    #meanC = np.mean(selected_coords_C, axis=0)
    #stdC = np.std(selected_coords_C, axis=0)
    #selected_coords_C = (selected_coords_C - meanC) / stdC



    #a = round(random.uniform(0.001, 0.999), 3)
    #b = round(random.uniform(0.001, 1 - a), 3)
    #c = round(1 - a - b, 3)

    numbers = [1,0,0]
    #numbers = [0.5,0.5,0]

    random.shuffle(numbers)
    numbers = np.array(numbers)
  
    Used_coords_A = np.random.choice(selected_coords_A.shape[0], size=int(numbers[0]*n), replace=False)
    Used_coords_B = np.random.choice(selected_coords_B.shape[0], size=int(numbers[1]*n), replace=False)
    Used_coords_C = np.random.choice(selected_coords_C.shape[0], size=int(numbers[2]*n), replace=False)




    combined_coords = np.concatenate((selected_coords_A[Used_coords_A], selected_coords_B[Used_coords_B], selected_coords_C[Used_coords_C]))
    return combined_coords,n,Lambda, len(Used_coords_A),len(Used_coords_B),len(Used_coords_C)

# Plot the samples
#import matplotlib.pyplot as plt

## Assign different colors and markers for each set
#colors = ['r', 'g', 'b']
#markers = ['o', 's', '^']
#labels = ['Set A', 'Set B', 'Set C']

#plt.xlim(combined_coords[:,0].min()-10000, combined_coords[:,0].max()+10000)
#plt.ylim(combined_coords[:,1].min()-10000, combined_coords[:,1].max()+10000)

#for i, coords in enumerate([selected_coords_A[Used_coords_A], selected_coords_B[Used_coords_B], selected_coords_C[Used_coords_C]]):
 #   plt.scatter(coords[:,0], coords[:,1], c=colors[i], marker=markers[i], label=labels[i])

#plt.legend()
#plt.show()

# Open a txt file for writing
M = 2000
for i in range(1000,1000+M):
    sample = generate_sample()
    coords = sample[0]
    n = sample[1]
    Lambda = sample[2]
    A = sample[3]
    B = sample[4]
    C = sample[5] 
    filename = r'C:\Users\Matth\OneDrive\Documenten\University\Master Thesis\Testdata2\sample_{}_{}_{}_{}_{}_{}.txt'.format(i, n, Lambda, A, B, C)
    with open(filename, "w") as f:
        
        # Write each coordinate with its index to the file in the desired format
        for i, (x, y) in enumerate(coords):
            f.write(f"{i+1} {int(x)} {int(y)}\n")