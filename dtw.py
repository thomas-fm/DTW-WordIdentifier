import pandas as pd
import numpy as np
from scipy.sparse import lil_matrix

def euclidian_dist(x, y):
    return abs(x - y)**2

'''
DTW Implementation
param : 2 series or array and distance function
return cost_matrix, path, minimum cost(distance)
'''
def dtw(s1, s2, dist):
    len_s1, len_s2 = len(s1), len(s2)
    cost_matrix = np.zeros((len_s1+1, len_s2+1))
    
    # Initialize cost_matrix
    for i in range(len_s1+1):
        for j in range(len_s2+1):
            cost_matrix[i, j] = np.inf
    cost_matrix[0, 0] = 0
    
    # Dynamic programming approaches
    for i in range(1, len_s1+1):
        for j in range(1, len_s2+1):
            cost = dist(s1[i-1], s2[j-1])
            prev = np.min([cost_matrix[i-1, j], cost_matrix[i, j-1], cost_matrix[i-1, j-1]])
            cost_matrix[i, j] = cost + prev

    # Construct path and cost
    idx_i, idx_j = len_s1, len_s2
    path = []   
    path.append(cost_matrix[idx_i, idx_j])

    while (idx_i != 0 and idx_j != 0):
        i, j = idx_i, idx_j
        min_prev = np.min([cost_matrix[i-1, j], cost_matrix[i, j-1], cost_matrix[i-1, j-1]])

        if (cost_matrix[i-1, j] == min_prev):
            idx_i, idx_j = i-1, j
        if (cost_matrix[i, j-1] == min_prev):
            idx_i, idx_j = i, j-1
        if (cost_matrix[i-1, j-1] == min_prev):
            idx_i, idx_j = i-1, j-1
        
        path.append(min_prev)
    
    # cost = cost_matrix[-1, -1]/(len_s1 + len_s2)
    cost = sum(path)/(len(path)-1)

    return cost_matrix[1:, 1:], path, cost

# Testing
# series1 = pd.Series([1,5,3,2,9])
# series2 = pd.Series([1,7,2,3,2,2,8])
# # series3 = pd.Series([1,2,2,1,0,1,1,2,1,2])
# # series4 = pd.Series([3,4,5,3,3,2,3,4,2,3])
# matrix,path,cost = dtw(series1, series2, euclidian_dist)
# print(matrix)
# print(path)