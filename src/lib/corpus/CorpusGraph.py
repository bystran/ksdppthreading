import networkx as nx
from src.lib.corpus.Corpus import Corpus


class CorpusGraph:
    def __init__(
        self,
        corpus: Corpus,
        cut_off_point = 0.1
    ):
        self.doc_graph = None 
        self.corpus = corpus
        self.cut_off_point = cut_off_point

    def create(self):
        G = nx.DiGraph()

        time_stamps = self.corpus.data_frame.time_stamp.values
        for i in range(len(self.corpus.data_frame)):
            time_stamp_i = time_stamps[i]
            for j in range(len(self.corpus.data_frame)):
                time_stamp_j = time_stamps[j]
                edge_weight = self.corpus.similarity_matrix[i][j]
                if i!=j and edge_weight > self.cut_off_point and time_stamp_i < time_stamp_j:
                    G.add_edge(i,j, weight=edge_weight)
                    
        nodes = set(G.nodes)
        for i in range(len(self.corpus.data_frame)):
            if i not in nodes:
                nodes.add(i)
                G.add_node(i)

        self.doc_graph = G
    
    def getAdjecencyMatrix(self):
        if self.doc_graph == None:
            self.create()
        
        return nx.to_numpy_array(self.doc_graph, nodelist=sorted(self.doc_graph.nodes()))