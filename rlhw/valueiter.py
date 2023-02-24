import numpy as np

def FLVI(map_name, g, slippery, tol=1e-12):
    mapping = {'S': 1, 'F': 1, 'G': 1, 'H': 0}
    if map_name == "4x4":
        grid = ["SFFF", "FHFH", "FFFH", "HFFG"]
    elif map_name == "8x8":
        grid = ["SFFFFFFF", "FFFFFFFF", "FFFHFFFF", "FFFFFHFF", "FFFHFFFF", "FHHFFFHF", "FHFFHFHF", "FFFHFFFG"]
    grid = np.vectorize(mapping.get)([list(s) for s in grid])
    V = np.zeros(grid.shape)
    N = grid.shape[0]
    V[-1,-1] = 1

    while True:
        Vnew = np.copy(V)
        for i in range(N):
            for j in range(N):
                if i+j == 2*(N-1) or grid[i,j] == 0:
                    continue
                left = g*V[i,max(0,j-1)]*grid[i,max(0,j-1)]
                up = g*V[max(0,i-1),j]*grid[max(0,i-1),j]
                if i+min(N-1,j+1) == 2*(N-1):
                    right = 1
                else:
                    right = g*V[i,min(N-1,j+1)]*grid[i,min(N-1,j+1)]
                if i+min(N-1,j+1) == 2*(N-1):
                    down = 1
                else:
                    down = g*V[min(i+1,N-1),j]*grid[min(i+1,N-1),j]
                if slippery:
                    Vnew[i,j] = max(up+left+right, down+left+right, left+up+down, right+up+down)/3
                else:
                    Vnew[i,j] = max(up, down, left, right)
        if np.max(np.abs(V-Vnew)) < tol:
            break
        V = Vnew
    V[-1,-1] = 0
    return V.flatten()
