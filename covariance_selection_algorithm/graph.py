from typing import List


class Graph:
    def __init__(self, n: int):
        self.__edges = [[] for i in range(n)]

    def add_edge(self, v: int, u: int, cost: float):
        self.__edges[v].append((u, cost))
        self.__edges[u].append((v, cost))

    def set_dists(self, v: int, dists: List[float], par: int = -1, cur_dist: float = 1):
        dists[v] = cur_dist
        for u, cost in self.__edges[v]:
            if u == par:
                continue
            self.set_dists(u, dists, v, cur_dist * cost)
