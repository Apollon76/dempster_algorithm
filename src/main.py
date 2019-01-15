import copy
from typing import Set, Tuple
from math import log, pi

import pandas as pd
import numpy as np


def calc_likelihood(sigma: np.ndarray) -> float:
    p = sigma.shape[0]

    # sigma = copy.deepcopy(sigma)
    # for i in range(p):
    #     for j in range(i):
    #         sigma[i, j] = sigma[j, i]

    try:
        return -p / 2 * log(2 * pi) - 1 / 2 * log(np.linalg.det(sigma)) - p / 2
    except ValueError:
        print(log(np.linalg.det(sigma)))
        print(sigma)
        raise


def get_sigma(ind: Tuple[int, int], s: np.ndarray, a: Set[Tuple[int, int]]) -> float:
    i, j = ind
    return s[i][j] if ind in a else 0


def get_gamma(ind1: Tuple[int, int], ind2: Tuple[int, int], sig: np.ndarray):
    i, j = ind1
    k, l = ind2

    if i != j:
        if k != l:
            return -(sig[i, k] * sig[j, l] + sig[i, l] * sig[j, k])
        else:
            return -sig[i, k] * sig[j, k]
    else:
        if k != l:
            return -sig[i, k] * sig[j, l]
        else:
            return -1 / 2 * sig[i, k] ** 2


def calc_next_estimation(edges: Set[Tuple[int, int]], sigma: np.ndarray, s: np.ndarray) -> np.ndarray:
    p = s.shape[0]

    inv_sigma = np.linalg.inv(sigma)
    indices = list(edges)

    theta = np.asarray([-s[i, j] if i != j else -1 / 2 * s[i, j] for i, j in indices])

    delta = float('inf')
    eps = 0.0001
    while delta > eps:
        gamma = np.ndarray(shape=(len(edges), len(edges)), dtype=float)

        for i, e1 in enumerate(indices):
            for j, e2 in enumerate(indices):
                gamma[i, j] = get_gamma(e1, e2, sigma)

        fa0 = np.asarray([inv_sigma[i, j] for i, j in indices])
        theta0 = np.asarray([-sigma[i, j] if i != j else -1 / 2 * sigma[i, j] for i, j in indices])

        # print(gamma)
        # print((theta - theta0))
        s = np.linalg.solve(gamma, (theta - theta0))
        new_fa = fa0 - s

        inv_sigma = np.zeros(shape=(p, p))
        for ind, e in enumerate(indices):
            i, j = e
            inv_sigma[i, j] = new_fa[ind]
            inv_sigma[j, i] = new_fa[ind]

        sigma = np.linalg.inv(inv_sigma)

        delta = np.dot(new_fa, new_fa) - np.dot(fa0, fa0)

    return sigma


def main():
    correlation_matrix = pd.read_csv('../TestData/DempsterExample/data.csv',
                                     dtype={i: float for i in range(6)},
                                     delimiter=';',
                                     names=[i for i in range(6)],
                                     skipinitialspace=True)
    correlation_matrix = correlation_matrix.values
    p = correlation_matrix.shape[0]
    for i in range(p):
        for j in range(i):
            correlation_matrix[i, j] = correlation_matrix[j, i]
    print(correlation_matrix)
    '''
    p = int(input())
    s = np.zeros(shape=(p, p))
    for i in range(p):
        cur = list(map(float, input().split()))
        for j in range(p):
            s[i, j] = cur[j]
        '''

    alpha = 0.5

    corr_estimation = np.identity(p)
    processed = set()
    for i in range(p):
        processed.add((i, i))

    cur_delta = float('inf')
    base_likelihood = calc_likelihood(corr_estimation)
    print(base_likelihood)
    k = 0
    while cur_delta >= alpha / (p * (p - 1) / 2 - k):
        k += 1
        max_delta = 0
        best_edge = None
        for i in range(p):
            for j in range(i, p):
                if (i, j) in processed:
                    continue

                cur_processed = copy.deepcopy(processed)
                cur_processed.add((i, j))
                new_corr_estimation = calc_next_estimation(cur_processed, corr_estimation, correlation_matrix)
                cur_likelihood = calc_likelihood(new_corr_estimation)
                cur_delta = cur_likelihood - base_likelihood
                # if i == 3 and j == 4:
                # print(cur_delta)
                if cur_delta > max_delta:
                    best_edge = (i, j)
                    max_delta = cur_delta
        if best_edge is None:
            break

        print(best_edge)
        processed.add(best_edge)
        corr_estimation = calc_next_estimation(processed, corr_estimation, correlation_matrix)

        cur_likelihood = calc_likelihood(corr_estimation)
        cur_delta = cur_likelihood - base_likelihood
        base_likelihood = cur_likelihood

    print(processed)
    print(corr_estimation)
    for i in range(p):
        for j in range(i):
            corr_estimation[i, j] = corr_estimation[j, i]
    inverted = np.linalg.inv(corr_estimation)
    for i in inverted:
        for j in i:
            print('%.4f' % j, end=' ')
        print()


if __name__ == '__main__':
    main()
