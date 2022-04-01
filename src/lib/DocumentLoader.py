import pandas as pd
from datetime import datetime

class DocumentLoader:
    def __init__(self):
        self.df = None
        
    def load_excluding_storyLen(self, file_path, story_size, top=None):
        df = pd.read_csv(file_path)
        df = df[df['text'].str.len() > 100]
        df = df[df['title'].str.len() > 10]
        counts = df[['story_id','doc_id']].groupby("story_id").count().reset_index().rename(columns={'doc_id': 'counts'})
        df = df.merge(counts, on='story_id')
        df = df[df['counts'] != story_size]
        df.insert(0, 'order', range(0, len(df)))
        df.reset_index(drop=True, inplace=True)
        df['time_stamp'] = df['date'].apply(self.datetime_to_time_stamp)
        if top is None:
            top = len(df)
        self.df = df.head(top)
          
    def load_full(self, file_path):
        df = pd.read_csv(file_path)
        df = df[df['text'].str.len() > 100]
        df.insert(0, 'order', range(0, len(df)))
        df.reset_index(drop=True, inplace=True)
        df['time_stamp'] = df['date'].apply(self.datetime_to_time_stamp)
        self.df = df

    def load_first_x_stories(self, file_path, up_to_story_id):
        df = pd.read_csv(file_path)
        df = df[df['story_id'] < up_to_story_id]
        df = df[df['text'].str.len() > 100]
        df.insert(0, 'order', range(0, len(df)))
        df.reset_index(drop=True, inplace=True)
        df['time_stamp'] = df['date'].apply(self.datetime_to_time_stamp)
        self.df = df
        
    def load_by_story_length(self, file_path, story_size, top=None):
        df = pd.read_csv(file_path)
        df = df[df['text'].str.len() > 100]
        df = df[df['title'].str.len() > 10]
        counts = df[['story_id','doc_id']].groupby("story_id").count().reset_index().rename(columns={'doc_id': 'counts'})
        df = df.merge(counts, on='story_id')
        df = df[df['counts'] == story_size]
        df.insert(0, 'order', range(0, len(df)))
        df.reset_index(drop=True, inplace=True)
        df['time_stamp'] = df['date'].apply(self.datetime_to_time_stamp)
        if top is None:
            top = len(df)
        self.df = df.head(top)
        
        
    def datetime_to_time_stamp(self, datetime_string):
        s = datetime_string.strip()
        return datetime.strptime(s, '%d/%m/%Y %H:%M').timestamp()
