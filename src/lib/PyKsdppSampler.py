import math
import matlab.engine
import numpy as np

from src.lib.corpus.KsdppCorpusInputData import KsdppCorpusInputData


class PyKsdppSampler:
    def __init__(
        self,         
    ):
        self.eng = matlab.engine.start_matlab()
        s = self.eng.genpath('src\kulezsa-dpp')
        self.eng.addpath(s, nargout=0)
        
    def process(
        self,
        thread_size, 
        input_corpus: KsdppCorpusInputData 
    ):
        self.model = {}
        number_of_nodes = len(input_corpus.saliance_scores)
        D = len(input_corpus.graph_node_features[0])    #similarity feature dimension  

        self.model["T"] = thread_size
        self.model["N"] = number_of_nodes
        self.model["Q"] = matlab.double(list(input_corpus.saliance_scores))

        print('Assinging A')
        self.model["A"] = matlab.double(input_corpus.relation_graph.tolist())

        print('Assinging G')
        #NxD similarity features - in the context of the paper this is 1000 most similar
        #documents array of size n projected into 50 
        self.model["G"] = matlab.double((input_corpus.graph_node_features+1).tolist())
        print('Calling bp')
        bp_cov = self.eng.bp(self.model, 'covariance')

        print('Calling decompose kernel')
        self.model['C'] = self.eng.decompose_kernel(bp_cov)
    
    def dppSample(
        self,
        num_of_threads
    ):
        print('Calling sdpp sample')
        sdpp_samples = self.eng.sample_sdpp(self.model, self.model['C'], num_of_threads)
        return np.asarray(sdpp_samples).astype(int).T - 1
    
    def sample(
        self,
        thread_size, 
        num_of_threads,
        input_corpus: KsdppCorpusInputData,         
    ):
        self.process(thread_size, input_corpus)
        return self.dppSample(num_of_threads)
        





        