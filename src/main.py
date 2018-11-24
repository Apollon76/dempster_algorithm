import copy
from typing import Set, Tuple
from math import log, pi

import pandas as pd
import numpy as np


def l(sigma: np.ndarray) -> float:
    p = sigma.shape[0]
    return -p / 2 * log(2 * pi) - 1 / 2 * log(np.linalg.det(sigma)) - p / 2


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


def calc_sigma(a: Set[Tuple[int, int]], sigma: np.ndarray, s: np.ndarray) -> np.ndarray:
    p = s.shape[0]

    inv_sigma = np.linalg.inv(sigma)
    indices = list(a)

    theta = np.asarray([-s[i, j] if i != j else -1 / 2 * s[i, j] for i, j in indices])
    # print(theta)

    gamma = np.ndarray(shape=(len(a), len(a)), dtype=float)

    '''
    partial_s = s.copy()
    for i in range(p):
        for j in range(p):
            if (i, j) in a:
                continue
            partial_s[i, j] = 0
            '''

    for i, e1 in enumerate(indices):
        for j, e2 in enumerate(indices):
            gamma[i, j] = get_gamma(e1, e2, sigma)

    #print(gamma)
    #print(np.linalg.inv(gamma))

    delta = float('inf')
    eps = 0.0001
    while delta > eps:
        fa0 = np.asarray([inv_sigma[i, j] for i, j in indices])
        theta0 = np.asarray([-sigma[i, j] if i != j else -1 / 2 * sigma[i, j] for i, j in indices])
        # print('theta0', theta0)
        #print('fa0', fa0)

        s = np.linalg.solve(gamma, (theta - theta0))
        #print(theta - theta0)
        new_fa = fa0 - s
        #print('new_fa', new_fa)

        inv_sigma = np.zeros(shape=(p, p))
        for ind, e in enumerate(indices):
            i, j = e
            inv_sigma[i, j] = new_fa[ind]

        sigma = np.linalg.inv(inv_sigma)

        delta = np.dot(new_fa, new_fa) - np.dot(fa0, fa0)
    return sigma


def main():
    s = pd.read_csv('../TestData/DempsterExample/data.csv',
                       dtype={i: float for i in range(6)},
                       delimiter=';',
                       names=[i for i in range(6)],
                       skipinitialspace=True)
    s = s.values
    print(s)
    p = 6
    '''
    p = int(input())
    s = np.zeros(shape=(p, p))
    for i in range(p):
        cur = list(map(float, input().split()))
        for j in range(p):
            s[i, j] = cur[j]
        '''

    alpha = 0.05

    sigma = np.identity(p)
    a = set()
    for i in range(p):
        a.add((i, i))

    g1 = float('inf')
    l0 = l(sigma)
    print(l0)
    while g1 >= alpha:
        g0 = 0
        best_edge = None
        for i in range(p):
            for j in range(i, p):
                if (i, j) in a:
                    continue
                a1 = copy.deepcopy(a)
                a1.add((i, j))
                # a1.add((3, 4))
                new_sigma = calc_sigma(a1, sigma, s)
                l1 = l(new_sigma)
                print(new_sigma)
                print(l1)
                g1 = l1 - l0
                if g1 > g0:
                    best_edge = (i, j)
                    g0 = g1
        print(best_edge)
        if best_edge is None:
            break

        a.add(best_edge)
        sigma = calc_sigma(a, sigma, s)

        l1 = l(sigma)
        g1 = l1 - l0
        l0 = l1
    print(a)


if __name__ == '__main__':
    main()

