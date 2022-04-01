from lexrank import STOPWORDS, LexRank
import numpy as np

class TextRanker:
    def __init__(self):
        pass
    
    def rank_documents(self, file_name, textSeries):
        print('ranking document')
        lxr = LexRank(textSeries.to_numpy(), stopwords=STOPWORDS['en'])
        self.ranks = lxr.rank_sentences(
            textSeries.values.astype('U'),
            threshold=None,
            fast_power_method=False,
        )
        
        self.save(file_name)
        
        return self.ranks
    
    
    def save(self, file_name):
        ranks = np.array(self.ranks)
        np.save(file_name, ranks)
    
    
    def load_or_rank(self, file_name, textSeries):
        try:
            self.ranks = np.load(file_name)
        except: 
            self.ranks = self.rank_documents(file_name, textSeries)
    
        return self.ranks


    

