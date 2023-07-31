import numpy as np

def mst(Dist_matrix):
    n = len(Dist_matrix)
    T_Dist_matrix = Dist_matrix.copy()
    for i in range(n):
        T_Dist_matrix[i, i] = np.inf

    visited = np.zeros(n, dtype=bool)
    parent = np.zeros(n, dtype=int)
    key = np.zeros(n, dtype=float)

    # Initialize the key values to infinity
    key.fill(np.inf)

    # The first node is the root
    parent[0] = -1
    key[0] = 0

    for i in range(n-1):
        # Find the minimum key value among the nodes that have not been visited yet
        u = np.argmin(key[~visited])
        visited[u] = True

        # Update the key values of the neighboring nodes
        for v in range(n):
            if not visited[v] and T_Dist_matrix[u, v] < key[v]:
                key[v] = T_Dist_matrix[u, v]
                parent[v] = u

    # Construct the MST edges from the parent array
    edges = []
    mst_length = 0
    for v in range(1, n):
        edges.append((parent[v], v))
        mst_length += Dist_matrix[v, parent[v]]




    return mst_length


