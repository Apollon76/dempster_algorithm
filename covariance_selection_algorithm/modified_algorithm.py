from typing import Set, Tuple

import numpy as np

from covariance_selection_algorithm.algorithm import calculate, get_corr_estimation
from covariance_selection_algorithm.dsu import DSU
from covariance_selection_algorithm.graph import Graph


def estimation_by_processed(processed: Set[Tuple[int, int]],
                            correlation_matrix: np.ndarray) -> np.ndarray:
    p = correlation_matrix.shape[0]

    g = Graph(p)
    for v, u in processed:
        if v == u:
            continue
        g.add_edge(v, u, correlation_matrix[v, u])

    dists = [0] * p
    g.set_dists(1, dists, -1, 1)

    result = np.zeros((p, p))
    for i in range(p):
        dists = [0] * p
        g.set_dists(i, dists, -1, 1)
        print(dists)
        for j in range(p):
            result[i, j] = dists[j]

    return result


def calculate_with_modification(correlation_matrix: np.ndarray,
                                significance_level: float) -> np.ndarray:
    p = correlation_matrix.shape[0]

    processed = set()
    for i in range(p):
        processed.add((i, i))

    edges = []
    for i in range(p):
        for j in range(i + 1, p):
            edges.append((abs(correlation_matrix[i, j]), (i, j)))
    edges.sort(reverse=True)

    s = DSU(p)
    for _, edge in edges:
        v, u = edge
        if s.get_parent(v) != s.get_parent(u):
            s.merge(v, u)
            processed.add(edge)

    corr_estimation = estimation_by_processed(processed, correlation_matrix)

    return get_corr_estimation(
        corr_estimation,
        processed,
        correlation_matrix,
        significance_level
    )
