import numpy as np

class WordMoverDistance:
    def __init__(
        self, 
        model,
        tokenizer
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.memo = {}
        
    def getSimilarityMatrix(self, texts):
        self.memo = {}
        res = np.zeros((len(texts), len(texts)))
        vect_texts = [self.tokenizer(text) for text in texts]
        for i in range(len(vect_texts)):
            if i % 100 == 0:
                print(f'{i} ducements done.')
            for j in range(len(vect_texts)):
                if i!=j:
                    res[i, j] = self.getWmd(i,j, vect_texts)
                    
        res[res == np.inf] = -1
        res[res == -1] = np.max(res)+0.01
        res = np.max(res) - res
        return res
                    
        
    def getWmd(self, index_i, index_j, data):
        res = 0.0
        if (index_i, index_j) in self.memo:
            res = self.memo[(index_i, index_j)]
        elif (index_j, index_i) in self.memo:
            res = self.memo[(index_j, index_i)]
        else:
            res = self.model.wmdistance(data[index_i], data[index_j])
            self.memo[(index_i, index_j)] = res
        
        
        return res