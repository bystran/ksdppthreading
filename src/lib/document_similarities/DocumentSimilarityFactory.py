from src.lib.document_similarities.CosineDocumnetSimilarity import CosineDocumentSimilarity
from src.lib.document_similarities.WMDSimilarity import WMDSimilarity





class DocumentSimilarityFactory:
    def getDocumentSimilarityObject(
        self,
        method, 
        embedding_mode
    ):
        if method == 'cosine':
            return CosineDocumentSimilarity(embedding_mode)
        
        if method == 'wmd':
            return WMDSimilarity(embedding_mode)