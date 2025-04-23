
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

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
        self.document_vectors = self.svd_matrix

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
        query_tfidf = self.vectorizer.transform([query])
        query_tfidf_array = query_tfidf.toarray().flatten()
        query_terms = {}
        feature_names = self.vectorizer.get_feature_names_out()
        
        for i, score in enumerate(query_tfidf_array):
            if score > 0:
                query_terms[feature_names[i]] = float(score)
        
        query_terms = dict(sorted(query_terms.items(), key=lambda x: x[1], reverse=True)[:5])
        
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(query_svd, self.svd_matrix).flatten()
        
        top_indices = similarities.argsort()[::-1][:top_n]
        results = []
        
        concepts = self.get_term_concept_matrix(n_terms=5)
        
        for idx in top_indices:
            row = self.rows[idx]
            score = similarities[idx]
            
            if score > 0:
                doc_tfidf = self.tfidf_matrix[idx].toarray().flatten()
                matched_terms = {}
                important_terms = {}
                
                for term in query_terms:
                    term_idx = np.where(feature_names == term)[0]
                    if len(term_idx) > 0:
                        term_idx = term_idx[0]
                        if doc_tfidf[term_idx] > 0:
                            matched_terms[term] = float(doc_tfidf[term_idx])
                
                doc_top_indices = np.argsort(doc_tfidf)[::-1][:10]
                for i in doc_top_indices:
                    term = feature_names[i]
                    if term not in matched_terms and doc_tfidf[i] > 0:
                        important_terms[term] = float(doc_tfidf[i])
                
                important_terms = dict(sorted(important_terms.items(), 
                                            key=lambda x: x[1], 
                                            reverse=True)[:5])
                
                doc_vector = self.svd_matrix[idx]
                top_concept_indices = np.argsort(np.abs(doc_vector))[::-1][:3]
                
                themes = []
                for concept_idx in top_concept_indices:
                    concept_name = f"Concept {concept_idx+1}"
                    concept_terms = [term for term, _ in concepts[concept_name][:3]]
                    themes.append(f"{concept_name}: {', '.join(concept_terms)}")
                
                doc_similarities = self.get_document_similarity().iloc[idx].sort_values(ascending=False)
                similar_docs = [
                    {
                        "document": doc_label,
                        "score": float(sim_score),
                        'embedding': self.document_vectors[idx].tolist()
                    }
                    for doc_label, sim_score in doc_similarities[1:3].items()
                ]
                
                results.append({
                    'document': self.doc_labels[idx],
                    'score': float(score),
                    'row': row,
                    'themes': themes,
                    'similar_documents': similar_docs,
                    'embedding': self.document_vectors[idx].tolist(),
                    'query_embedding': query_svd.tolist()
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