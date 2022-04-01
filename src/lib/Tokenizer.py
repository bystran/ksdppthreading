import nltk
import spacy

class Tokenizer:
    def __init__(self):
        nlp = spacy.load('en_core_web_sm', disable=['ner'])
        nlp.remove_pipe('tagger')
        nlp.remove_pipe('parser')
        nltk.download('stopwords')
        self.nlp = nlp

    def spacy_tokenize(self, string):
        tokens = list()
        doc = self.nlp(string)
        for token in doc:
            tokens.append(token)
        return tokens

    def normalize(self, tokens, keep_stop_words = False):
        normalized_tokens = list()
        for token in tokens:
            normalized = token.text.lower().strip()
            if ((token.is_alpha or token.is_digit) and (token.is_stop == False or keep_stop_words)):
                normalized_tokens.append(normalized)
        return normalized_tokens
    
    def lemmatize(self, tokens):
        result = list()
        for token in tokens:
            if ((token.is_alpha or token.is_digit) and token.is_stop == False):
                result.append(token.lemma_)
        return result
            
    def tokenize_lemmatize(self, string):
        return self.lemmatize(self.spacy_tokenize(string))

    def tokenize_normalize(self,string):
        return self.normalize(self.spacy_tokenize(string))
    
    def tokenize_normalize_keep_stop_words(self, string):
        return self.normalize(self.spacy_tokenize(string), keep_stop_words = True)
