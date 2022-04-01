from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
import numpy as np


class CorpusFeatures:
    def __init__(
        self,
        feature_size = 1000,
        projected_features_size = 50
    ):
        self.feature_size = feature_size
        self.projected_features_size = projected_features_size
    
    def getNodeFeatures(self, corpus):
        feature_matrix = np.zeros((len(corpus.data_frame),len(corpus.data_frame)))
        for i in range(len(corpus.data_frame)):
            most_similar_indexes = np.argsort(corpus.similarity_matrix[i,:])[::-1][1:self.feature_size+1]
            feature_matrix[i,most_similar_indexes] = 1

        return feature_matrix

    def getProjectedNodeFeatures(self, corpus):
        node_features = self.getNodeFeatures(corpus)
        projection_transformer = GaussianRandomProjection(n_components=self.projected_features_size, random_state = 30)
        projected_node_features = projection_transformer.fit_transform(node_features)
        return projected_node_features
    
    def getPCANodeFeatures(self, corpus):
        node_features = self.getNodeFeatures(corpus)
        projection_transformer = PCA(n_components=self.projected_features_size)
        projected_node_features = projection_transformer.fit_transform(node_features)
        return projected_node_features
