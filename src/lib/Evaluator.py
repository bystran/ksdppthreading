from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import homogeneity_completeness_v_measure
from datetime import date
import numpy as np
import pandas as pd


class Evaluator:
    def __init__(self, corpus):
        self.corpus = corpus 
    
    def calculateMetrics(self, labels, pred):
        self.ari = adjusted_rand_score(labels, pred)
        self.ami = adjusted_mutual_info_score(labels, pred)
        self.nmi = normalized_mutual_info_score(labels,pred)
        self.h, self.c, self.v = homogeneity_completeness_v_measure(labels,pred)

    def calculateAverageRunMetrics(self, sampler, num_runs):
        ari = []
        ami = []
        nmi =[] 
        h = []
        c = []
        v = []
        
        for i in range(num_runs):
            threads_indexes = sampler.dppSample(5)
            lables = self.corpus.data_frame.iloc[threads_indexes.flatten()].story_id.to_numpy()
            pred = [ x // threads_indexes.shape[1] for x in range(threads_indexes.size) ]  
            self.calculateMetrics(lables, pred)
            ari.append(self.ari)
            ami.append(self.ami)
            nmi.append(self.nmi)
            h.append(self.h)
            c.append(self.c)
            v.append(self.v)
            
        self.ari = np.mean(ari)
        self.ami = np.mean(ami)
        self.nmi = np.mean(nmi)
        self.h = np.mean(h)
        self.c = np.mean(c)
        self.v = np.mean(v)
    def calculateAverageRunMetricsFullSmapler(self, sampler, data, num_runs):
        ari = []
        ami = []
        nmi =[] 
        h = []
        c = []
        v = []
        
        for i in range(num_runs):
            threads_indexes = sampler.sample(5, 5, data)
            lables = self.corpus.data_frame.iloc[threads_indexes.flatten()].story_id.to_numpy()
            pred = [ x // threads_indexes.shape[1] for x in range(threads_indexes.size) ]  
            self.calculateMetrics(lables, pred)
            ari.append(self.ari)
            ami.append(self.ami)
            nmi.append(self.nmi)
            h.append(self.h)
            c.append(self.c)
            v.append(self.v)
            
        self.ari = np.mean(ari)
        self.ami = np.mean(ami)
        self.nmi = np.mean(nmi)
        self.h = np.mean(h)
        self.c = np.mean(c)
        self.v = np.mean(v)
    
    
    def printMetrics(self):
        print(f"ARI: {self.ari}")
        print(f"AMI: {self.ami}")
        print(f"NMI: {self.nmi}")
        print(f"H  : {self.h}")
        print(f"C  : {self.c}")
        print(f"V  : {self.v}")
    
    def to_json(self):
        return {
            'ari': self.ari,
            'ami': self.ami,
            'nmi': self.nmi,
            'h': self.h,
            'c': self.c,
            'v': self.v
        }
        
    def saveToFile(
        self,
        file_name, 
        model_name,
        data_name, 
        algorithm = 'ksdpp',  
    ):
        column_names = ["date", "model_name", "data_name", "algorithm",
                        "ari", "ami", "nmi", "h", "c", "v"]
        try:
            df = pd.read_csv(file_name);
        except:

            df = pd.DataFrame(columns = column_names)
        
        newData = [
            str(date.today()),
            model_name,
            data_name,
            algorithm, 
            self.ari, self.ami, self.nmi,
            self.h, self.c, self.v
        ]

        newDataDf = pd.DataFrame([newData], columns = column_names)
        df = df.append(newDataDf)
        
        df.to_csv(file_name)
        