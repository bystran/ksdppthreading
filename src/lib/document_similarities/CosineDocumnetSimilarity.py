
from numpy import vectorize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import os.path
from src.lib.Tokenizer import Tokenizer
from src.lib.vectorizers.BertVectorizer import BertVectorizer
from src.lib.vectorizers.Word2VecVectorizer import Word2VecVectorizer
import gensim.downloader as api
from easybert import Bert
import numpy as np

class CosineDocumentSimilarity:
    def __init__(
        self,
        embedding_mode
    ):
        tokenizer = Tokenizer()
        
        if embedding_mode == 'tfidf':
            self.vectorizer = TfidfVectorizer(tokenizer=tokenizer.tokenize_normalize)
        
        if embedding_mode.startswith('word2vec'):
            mode = embedding_mode.split('-')[-1]
            model = api.load("word2vec-google-news-300")
            vector_size = 300
            if mode == 'minmax':
                vector_size*=2
            
            self.vectorizer = Word2VecVectorizer(
                tokenizer=tokenizer.tokenize_normalize,
                model=model,
                vector_size=vector_size,
                mode=mode    
            )
            
        if embedding_mode.startswith('glove'):
            mode = embedding_mode.split('-')[-1]
            model = api.load('glove-wiki-gigaword-300')
            vector_size = 300
            if mode == 'minmax':
                vector_size*=2
            
            self.vectorizer = Word2VecVectorizer(
                tokenizer=tokenizer.tokenize_normalize,
                model=model,
                vector_size=vector_size,
                mode=mode    
            )
            
        if embedding_mode == 'bert':
            model = Bert("https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1")
            self.vectorizer = BertVectorizer(
                tokenizer=tokenizer.tokenize_normalize,
                model=model,
                vector_size=768
            )
        
        
    def getSimilarityMatrix(self, document_texts, cache_file_name=None):
        
        if cache_file_name is not None and os.path.exists(cache_file_name):
            similarity_matrix = np.load(cache_file_name)
        else:
            doc_vectors = self.vectorizer.fit_transform(document_texts)
            similarity_matrix = cosine_similarity(doc_vectors)
            if cache_file_name is not None:
                np.save(cache_file_name, similarity_matrix)

        return similarity_matrix
