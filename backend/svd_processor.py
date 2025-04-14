
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class SVDTextProcessor:
    def __init__(self, rows, n_components=100, weight_fields=None, weight_factor=5.0):
        self.rows = rows
        self.weight_fields = weight_fields or ['Name of Incident', 'description', 'reddit_posts']
        self.weight_factor = weight_factor
        self.n_components = min(n_components, len(rows) - 1)  # SVD components cannot exceed n_docs - 1
        
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            max_features=5000,
            token_pattern=r'\b[a-zA-Z]{3,}\b'
        )
        
        self.svd = TruncatedSVD(n_components=self.n_components)        
        self.normalizer = Normalizer(copy=False)
        self.pipeline = make_pipeline(self.svd, self.normalizer)        
        self.corpus = self._prepare_weighted_corpus()
        
        self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus)
        self.svd_matrix = self.pipeline.fit_transform(self.tfidf_matrix)        
        self.feature_names = self.vectorizer.get_feature_names_out()
            
        self.doc_labels = [
            f"{row.get('Name of Incident', 'Unknown')} ({row.get('Place Name', 'Unknown')})"
            for row in self.rows
        ]
        
        self.explained_variance = self.svd.explained_variance_ratio_.sum()
    
    def _prepare_weighted_corpus(self):
        corpus = []
        
        for row in self.rows:
            document_parts = []            
            for key, value in row.items():
                if value and isinstance(value, (str, int, float)):
                    document_parts.append(str(value))
            
            for field in self.weight_fields:
                repetitions = int(self.weight_factor) - 1
                for _ in range(repetitions):
                    if field in row and row[field]:
                        document_parts.append(str(row[field]))
            
            corpus.append(" ".join(document_parts))
        
        return corpus
    
    def get_document_similarity(self):
        from sklearn.metrics.pairwise import cosine_similarity
        
        similarity_matrix = cosine_similarity(self.svd_matrix)
        similarity_df = pd.DataFrame(similarity_matrix, index=self.doc_labels, columns=self.doc_labels)
        
        return similarity_df
    
    def search(self, query, top_n=5):
        query_vector = self.vectorizer.transform([query])        
        query_svd = self.pipeline.transform(query_vector)        
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(query_svd, self.svd_matrix).flatten()
        
        top_indices = similarities.argsort()[::-1][:top_n]
        results = []
        for idx in top_indices:
            row = self.rows[idx]
            score = similarities[idx]
            if score > 0:
                results.append({
                    'document': self.doc_labels[idx],
                    'score': score,
                    'row': row
                })
        
        return results
    
    def get_term_concept_matrix(self, n_terms=20):
        term_concept_matrix = self.svd.components_
        
        concepts = {}
        for i, concept_vec in enumerate(term_concept_matrix):
            sorted_indices = np.argsort(np.abs(concept_vec))[::-1][:n_terms]
            terms = [(self.feature_names[idx], concept_vec[idx]) for idx in sorted_indices]
            
            concepts[f"Concept {i+1}"] = terms
            
        return concepts
    
    def get_document_vectors(self):
        return self.svd_matrix
    
    def get_topic_strength(self, doc_idx):
        doc_vector = self.svd_matrix[doc_idx]
        topics = {f"Topic {i+1}": strength for i, strength in enumerate(doc_vector)}
        
        return topics
    
    def get_similar_documents(self, doc_idx, top_n=5):
        from sklearn.metrics.pairwise import cosine_similarity
        
        doc_vector = self.svd_matrix[doc_idx].reshape(1, -1)
        similarities = cosine_similarity(doc_vector, self.svd_matrix).flatten()        
        similarities[doc_idx] = -1
        
        top_indices = similarities.argsort()[::-1][:top_n]
        similar_docs = []
        
        for idx in top_indices:
            if similarities[idx] > 0:
                similar_docs.append({
                    'document': self.doc_labels[idx],
                    'score': similarities[idx],
                    'row': self.rows[idx]
                })
        
        return similar_docs