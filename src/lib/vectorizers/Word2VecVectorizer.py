from sklearn.base import TransformerMixin
import numpy as np

class Word2VecVectorizer(TransformerMixin):
    def __init__(
        self,
        tokenizer,
        model,
        vector_size,
        mode='mean'
    ):
        self.model = model
        self.vector_size = vector_size
        self.tokenizer = tokenizer
        self.mode = mode
    
    def fit_transform(self, document_texts):
        return self.embedDocuments(document_texts)
    
    def embedDocuments(self, document_texts):
        res = np.zeros((len(document_texts), self.vector_size))
        for i in range(len(document_texts)):
            if self.mode == 'minmax':
                res[i] = self.getMinMaxSentanceEmbeding(document_texts[i])
            else:
                res[i] = self.getSentanceEmbeding(document_texts[i])
        return res
    
    def getSentanceEmbeding(self, sentence):
        tokens = self.tokenizer(sentence)
        if(len(tokens) == 0):
            return np.zeros((1, self.vector_size))
        res = np.zeros((len(tokens), self.vector_size))
        for i in range(len(tokens)):
            if(tokens[i] in self.model.key_to_index):
                res[i] = self.model.get_vector(tokens[i])

        return np.mean(res, axis=0)
    def getMinMaxSentanceEmbeding(self, sentence):
        tokens = self. tokenizer(sentence)
        if(len(tokens) == 0):
            return np.zeros((1, self.vector_size))
        embeddings = np.zeros((len(tokens), self.vector_size//2))
        for i in range(len(tokens)):
            if(tokens[i] in self.model.key_to_index):
                embeddings[i] = self.model.get_vector(tokens[i])
        element_wise_min = np.min(embeddings, axis=0)
        element_wise_max = np.max(embeddings, axis=0)
        
        return np.concatenate((element_wise_min, element_wise_max), axis=None)
