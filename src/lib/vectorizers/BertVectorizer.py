import warnings
import logging, sys
logging.disable(sys.maxsize)
warnings.filterwarnings('ignore')

import numpy as np



from sklearn.base import TransformerMixin

class BertVectorizer(TransformerMixin):
    def __init__(
        self,
        tokenizer,
        model,
        vector_size
    ):
        self.model = model
        self.vector_size = vector_size
        self.tokenizer = tokenizer
    
    def fit_transform(self, document_texts):
        model = self.model
        with model:
            return self.embedDocuments(document_texts, model)
    
    def embedDocuments(self, document_texts, model):
        res = np.zeros((len(document_texts), self.vector_size))
        for i in range(len(document_texts)):
            if i%50 == 0:
                print(f"{i}/{len(document_texts)}", end='\r')
            res[i] = self.getMeanSentanceEmbedding(document_texts[i], model)
        return res
    
    def getSentanceEmbeding(self, sentence, model):
        tokens = self.tokenizer(sentence)
        res = model.embed(sentence, per_token=False)

        return np.mean(res, axis=0)
    
    def getMeanSentanceEmbedding(self, sentence, model):
        res = model.embed(sentence, per_token=True)
        return np.mean(res, axis=0)
