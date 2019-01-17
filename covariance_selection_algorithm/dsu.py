class DSU:
    def __init__(self, n: int):
        self.__weights = [1] * n
        self.__par = [i for i in range(n)]

    def get_parent(self, v: int) -> int:
        if self.__par[v] == v:
            return v
        return self.get_parent(self.__par[v])

    def merge(self, v: int, u: int):
        v = self.get_parent(v)
        u = self.get_parent(u)
        if v == u:
            return
        if self.__weights[v] < self.__weights[u]:
            v, u = u, v
        self.__par[u] = v
        self.__weights[v] += self.__weights[u]
