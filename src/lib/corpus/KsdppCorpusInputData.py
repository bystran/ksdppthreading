class KsdppCorpusInputData:
    def __init__(
        self,
        relation_graph,
        saliance_scores,
        graph_node_features,
    ):
        self.relation_graph = relation_graph
        self.saliance_scores = saliance_scores
        self.graph_node_features = graph_node_features
    