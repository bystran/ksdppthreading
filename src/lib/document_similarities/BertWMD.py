from ast import excepthandler
import numpy as np
from gensim.models import KeyedVectors
import warnings
import logging, sys
logging.disable(sys.maxsize)
warnings.filterwarnings('ignore')
from easybert import Bert


class BertWMD:
    def __init__(
        self,
        tokenizer,
        model_url="https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1",
        
    ):
        self.bert = Bert(model_url)
        self.tokenizer = tokenizer
        self.kv = KeyedVectors(768)
   
    
    def build_keyed_vector(self, texts, bert):
        self.kv = KeyedVectors(768)
        self.pseudo_docs = []
        for i, text in enumerate(texts):
            print(f'{i}/{len(texts)} ducement embeddings done', end='\r')
            tokens = self.tokenizer(text)
            doc_vectors = bert.embed(tokens)
            doc_sentance = []
            for j in range(len(doc_vectors)):
                key_id = f"{i}-{j}"
                self.kv.add_vector(key_id, doc_vectors[j])
                doc_sentance.append(key_id)
            self.pseudo_docs.append(doc_sentance)
    

    
    def getSimilarityMatrix(self, texts):
        bert = self.bert
        self.memo = {}
        res = np.zeros((len(texts), len(texts)))
        with bert:
            self.build_keyed_vector(texts, bert)
            for i in range(len(self.pseudo_docs)):
                if i % 10 == 0:
                    print(f'{i}/{len(texts)} ducement distances done', end='\r')
                for j in range(len(self.pseudo_docs)):
                    if i!=j:
                        res[i, j] = self.getWmd(i,j, self.pseudo_docs, bert)
                    
        res[res == np.inf] = -1
        res[res == -1] = np.max(res)+0.01
        res = np.max(res) - res
        return res
                    
        
    def getWmd(self, index_i, index_j, data, bert):
        res = 0.0
        if (index_i, index_j) in self.memo:
            res = self.memo[(index_i, index_j)]
        elif (index_j, index_i) in self.memo:
            res = self.memo[(index_j, index_i)]
        else:
            try:
                res = self.kv.wmdistance(data[index_i], data[index_j], bert)
            except Exception as e:
                print(f'Failed to process distance between {index_i} and {index_j}')
                res = np.inf
            self.memo[(index_i, index_j)] = res
            
        return res