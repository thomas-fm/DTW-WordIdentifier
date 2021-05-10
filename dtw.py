import pandas as pd
import numpy as np
from scipy.sparse import lil_matrix

series1 = pd.Series([3, 4, 5, 3, 3])
series2 = pd.Series([1,2,2,1,0])
series3 = pd.Series([1,2,2,1,0,1,1,2,1,2])
series4 = pd.Series([3,4,5,3,3,2,3,4,2,3])

'''
Simple DTW Implementation
'''
def dtw(s1, s2):
    len_s1, len_s2 = len(s1), len(s2)
    cost_matrix = np.zeros((len_s1+1, len_s2+1))

    for i in range(len_s1+1):
        for j in range(len_s2+1):
            cost_matrix[i, j] = np.inf
    cost_matrix[0, 0] = 0
    
    # Dynamic programming approaches
    for i in range(1, len_s1+1):
        for j in range(1, len_s2+1):
            cost = abs(s1[i-1] - s2[j-1])**2
            #take last min from the window
            prev = np.min([cost_matrix[i-1, j], cost_matrix[i, j-1], cost_matrix[i-1, j-1]])
            cost_matrix[i, j] = cost + prev


    # Construct path

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

    return cost_matrix, path

'''
Improving DTW with Spares Matrix DP approach
'''
# dist = numpy.linalg.norm(a-b)

# Function to quantize the original data series
def quantize(s):
    return [(s[i] - np.min(s)) / (np.max(s) - np.min(s)) for i in range (len(s))]

# print(quantize(series1))
def coord_lower_neighbour(x, y, len_s1, len_s2):
    if (x!=0 and y!=0):
        coor1 = (x, y-1) if x>=0 and y-1>=0 else None
        coor2 = (x-1, y) if x-1>=0 and y>=0 else None
        coor3 = (x-1, y-1) if x-1>=0 and y-1>=0 else None
        return [n for n in (coor1, coor2, coor3) if n != None]
    return []

def coord_upper_neighbour(x, y, len_s1, len_s2):
    if (x!=len_s1-1 and y!=len_s2-1):
        coor1 = (x, y+1) if x<len_s1 and y+1<len_s2 else None
        coor2 = (x+1, y) if x+1<len_s1 and y<len_s2 else None
        coor3 = (x+1, y+1) if x+1<len_s1 and y+1<len_s2 else None
        return [n for n in (coor1, coor2, coor3) if n != None]
    return []
def sparse_dtw(s1, s2, res=0.5):
    s, q = quantize(s1), quantize(s2)
    len_s, len_q = len(s), len(q)
    sparse_matrix = lil_matrix((len_s, len_q), dtype=np.float64)

    # for i in range(len_s):
    #     for j in range(len_q):
    #         cost_matrix[i, j] = 0

    lower_bound = 0
    upper_bound = res

    # Fill the sparse matrix
    while 0 <= lower_bound and lower_bound <= 1 - res/2:
        # Search for each range of S and Q
        idxS = [i for i in range(len_s) if (lower_bound <= s[i] and s[i] <= upper_bound)]
        idxQ = [i for i in range(len_q) if (lower_bound <= q[i] and q[i] <= upper_bound)]

        lower_bound += +res/2
        upper_bound = lower_bound + res

        for i in idxS:
            for j in idxQ:
                dist = np.abs(s[i] - q[j])**2
                if dist == 0:
                    sparse_matrix[i, j] = -1
                else:
                    if sparse_matrix[i, j] != -1:
                        sparse_matrix[i,j]=sparse_matrix[i,j]+dist
                    else:
                        sparse_matrix[i,j]=dist
    # return sparse_matrix
    # Calculate cost for each element on sparse matrix

    for i in range(len_s):
        for j in range(len_q):
            if sparse_matrix[i,j] != 0:
                lower_n = [sparse_matrix[coord] for coord in coord_lower_neighbour(i, j, len_s, len_q) if sparse_matrix[coord] != 0]

                if lower_n:
                    prev_min = np.min(lower_n)
                    if prev_min == -1:
                        prev_min = 0
                else:
                    prev_min = 0

                if sparse_matrix[i,j]>-1:
                    sparse_matrix[i,j]=sparse_matrix[i,j]+prev_min
                elif sparse_matrix[i,j]==0:
                    pass
                elif sparse_matrix[i,j]==-1:
                    sparse_matrix[i,j]=prev_min if prev_min > 0 else -1
                
                upper_n = coord_upper_neighbour(i,j,len_s, len_q)
                if upper_n and not any(sparse_matrix[x,y] != 0 for x,y in upper_n):
                    for coor in upper_n:
                        if (sparse_matrix[coor] == 0):
                            _x, _y = coor
                            sparse_matrix[coor] = np.abs(s[_x] - q[_y])**2

    return sparse_matrix
matrix = sparse_dtw(series1, series2)
print(matrix)
# dtw_matrix, path = dtw(series4,series3)

# print(dtw_matrix)
# print(path)
# c = 10
# C = 11
# print(c)
# print(C)

