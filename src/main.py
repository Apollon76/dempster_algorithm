import copy
from typing import Set, Tuple

import numpy as np


def l(sigma: np.ndarray) -> float:
    ...


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


def calc_sigma(a: Set[Tuple[int, int]], sigma: np.ndarray, p: int, s: np.ndarray) -> np.ndarray:
    indices = list(a)

    theta = np.asarray([-s[i, j] if i != j else -1 / 2 * s[i, j] for i, j in indices])

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

    delta = float('inf')
    eps = 0.0001
    while delta > eps:
        fa0 = np.asarray([sigma[i, j] for i, j in indices])
        theta0 = np.asarray([-sigma[i, j] if i != j else -1 / 2 * sigma[i, j] for i, j in indices])

        s = np.linalg.solve(gamma, (theta - theta0))
        new_fa = fa0 + s

        inv_sigma = np.zeros(shape=(p, p))
        for ind, e in enumerate(indices):
            i, j = e
            inv_sigma[i, j] = new_fa[ind]

        sigma = np.linalg.inv(inv_sigma)

        delta = np.dot(new_fa, new_fa) - np.dot(fa0, fa0)
    return sigma


def main():
    n = int(input())
    p = int(input())
    s = np.zeros(shape=(p, p))
    for i in range(p):
        cur = list(map(float, input().split()))
        for j in range(p):
            s[i, j] = cur[j]

    alpha = 0.05

    sigma = np.identity(p)
    a = set()
    for i in range(p):
        a.add((i, i))

    g1 = float('inf')
    l0 = l(sigma)
    while g1 >= alpha:
        g0 = 0
        best_edge = None
        for i in range(p):
            for j in range(i, p):
                if (i, j) in a:
                    continue
                a1 = copy.deepcopy(a)
                a1.add((i, j))
                new_sigma = calc_sigma(a1)
                l1 = l(new_sigma)
                g1 = l1 - l0
                if g1 > g0:
                    best_edge = (i, j)
                    g0 = g1
        if best_edge is None:
            break

        a.add(best_edge)
        sigma = calc_sigma(a)

        l1 = l(sigma)
        g1 = l1 - l0
        l0 = l1
    print(a)


if __name__ == '__main__':
    main()

