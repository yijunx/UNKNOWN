class Graph:
    def __init__(self, dic=None):
        if dic is None:
            self.gdic = []
        self.gdic = dic

    def getVertices(self):
        return list(self.gdic.key())

    def getEdges(self):
        edges = []
        for ver in self.gdic:
            for another_ver in self.gdic[ver]:
                one_edge = {ver, another_ver}
                if one_edge not in edges:
                    edges.append(one_edge)
        return edges

