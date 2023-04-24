from math import exp

import numpy as np
import os

threshold = 0.000000001
c = 0.9


class Node:
    def __init__(self):
        pass


def converge(s, prev_s):
    if prev_s is None:
        return False
    distance_sum = 0
    for i in range(0, len(s)):
        for j in range(0, len(s)):
            distance_sum += abs(s[i][j] - prev_s[i][j])
    if distance_sum < threshold:
        return True
    else:
        return False


def get_in_neighbors(adjacency_matrix, i):
    res = []
    for j in range(0, len(adjacency_matrix)):
        if adjacency_matrix[j][i] > 0:
            res.append(j)
    return res


def ascos_plus_plus_weighted_sum(adjacency_matrix, s, i, j):
    weighted_sum = 0
    in_neighbors = get_in_neighbors(adjacency_matrix, i)
    w_i = 0
    for k in in_neighbors:
        w_i += adjacency_matrix[i][k]
    for k in in_neighbors:
        w_ik = adjacency_matrix[i][k]
        weighted_sum += (w_ik / w_i) * (1 - exp(-w_ik)) * s[k][j]
    return weighted_sum


def compute_ascos_plus_plus_scores(adjacency_matrix, c):
    n = len(adjacency_matrix)
    s = np.random.rand(n, n)
    prev_s = None
    while not converge(s, prev_s):
        prev_s = s.copy()
        for i in range(0, n):
            for j in range(0, n):
                if i != j:
                    s[i][j] = c * \
                        ascos_plus_plus_weighted_sum(adjacency_matrix, s, i, j)
                else:
                    s[i][j] = 1
    return s


def compute_similar_using_ascos_plus_plus(sssm, n, m, path):
    adjacency_matrix = np.ndarray((len(sssm), len(sssm)))

    for i in range(0, len(sssm)):
        n_similar_indices = np.flip(np.argsort(sssm[i]))[0:n]
        for j in range(0, len(sssm)):
            if j in n_similar_indices:
                adjacency_matrix[i][j] = sssm[i][j]
            else:
                adjacency_matrix[i][j] = 0
    s = compute_ascos_plus_plus_scores(adjacency_matrix, c)
    file = os.path.join(path, "task-8-output")
    os.makedirs(file, exist_ok=True)
    filename = os.path.join(file, "subject-subject-similarity-graph.txt")
    np.savetxt(filename, s, delimiter=', ', fmt='%.6f')
    s_t = s.T
    sum_j = np.zeros(len(s_t))
    for j in range(0, len(s_t)):
        sum_j[j] = np.sum(s_t[j])
    return np.flip(np.argsort(sum_j))[0:m]


def ascos_weighted_sum(adjacency_matrix, s, i, j):
    weighted_sum = 0
    in_neighbors = get_in_neighbors(adjacency_matrix, i)
    for k in in_neighbors:
        weighted_sum += s[k][j]
    return weighted_sum / len(in_neighbors)


def compute_ascos_scores(adjacency_matrix, c):
    n = len(adjacency_matrix)
    s = np.random.rand(n, n)
    prev_s = None
    while not converge(s, prev_s):
        prev_s = s.copy()
        for i in range(0, n):
            for j in range(0, n):
                if i != j:
                    s[i][j] = c * ascos_weighted_sum(adjacency_matrix, s, i, j)
                else:
                    s[i][j] = 1
    return s
