import numpy as np

from src.lib.Tokenizer import Tokenizer
from src.lib.document_similarities.BertWMD import BertWMD
import os.path

class WMDSimilarity:
    def __init__(
        self, 
        embedding_mode
    ):
        tokenizer = Tokenizer()
        if embedding_mode == 'bert':
            self.sim_calculator = BertWMD(tokenizer.tokenize_normalize_keep_stop_words)
            
    def getSimilarityMatrix(self, document_texts, cache_file_name=None):
        
        if cache_file_name is not None and os.path.exists(cache_file_name):
            similarity_matrix = np.load(cache_file_name)
        else:
            similarity_matrix = self.sim_calculator.getSimilarityMatrix(document_texts)
            if cache_file_name is not None:
                np.save(cache_file_name, similarity_matrix)

        return similarity_matrix
