


from cgitb import text
from src.lib.DocumentLoader import DocumentLoader
from src.lib.Evaluator import Evaluator
from src.lib.PyKsdppSampler import PyKsdppSampler
from src.lib.TextRanker import TextRanker
from src.lib.Tokenizer import Tokenizer
from src.lib.corpus.Corpus import Corpus
from src.lib.corpus.CorpusFeatures import CorpusFeatures
from src.lib.corpus.CorpusGraph import CorpusGraph
from src.lib.corpus.KsdppCorpusInputData import KsdppCorpusInputData
from src.lib.document_similarities.DocumentSimilarityFactory import DocumentSimilarityFactory

import time


class SemanticThreading:
    def __init__(self) -> None:
        return None
    
    def run(self) -> None:
        loader = DocumentLoader()
        loader.load_by_story_length('C:/Users/bysad/Programing/uni/semantic_threading/data/news_head_data.csv', 6)

        df = loader.df
        
        text_series = df['title']
        
        
        textRanker = TextRanker()
        text_lex_ranks = textRanker.load_or_rank("C:/Users/bysad/Programing/uni/semantic_threading/data/ranks/lexranks_story_size_6_all.npy", text_series)
        
        sim_factory = DocumentSimilarityFactory()
        similarity_calculator = sim_factory.getDocumentSimilarityObject('wmd', 'bert')

        similarity_matrix = similarity_calculator.getSimilarityMatrix(text_series.to_numpy(), "C:/Users/bysad/Programing/uni/semantic_threading/data/cosine_sims/wmd_bert_all.npy")

        # text_lex_ranks = text_lex_ranks /10000000000
        
        corpus = Corpus(df, similarity_matrix)
        
        graph = CorpusGraph(corpus, 0.69)
        adj_matrix = graph.getAdjecencyMatrix()


        features = CorpusFeatures(600, 50)    
        projected_node_features = features.getProjectedNodeFeatures(corpus)
        print(f'length: {len(adj_matrix[adj_matrix != 0])}')
        ksdpp_data = KsdppCorpusInputData(
            adj_matrix,
            text_lex_ranks,
            projected_node_features,
        )
        
        sampler = PyKsdppSampler()
        threads_indexes = sampler.sample(5, 5, ksdpp_data)
        
        
        labels = corpus.data_frame.iloc[threads_indexes.flatten()].story_id.to_numpy()
        pred = [ x // threads_indexes.shape[1] for x in range(threads_indexes.size) ]   

        
        evaluator = Evaluator(corpus)
        evaluator.calculateMetrics(labels, pred)
        evaluator.printMetrics()
        print(corpus.data_frame.iloc[threads_indexes.flatten()].title.to_numpy().reshape((5,5)))
        print(corpus.data_frame.iloc[threads_indexes.flatten()].story_id.to_numpy().reshape((5,5)))
        
        return corpus.data_frame.iloc[threads_indexes.flatten()].doc_id.to_numpy()




        
    
    