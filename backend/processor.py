from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd

class WeightedTfidfProcessor:
    def __init__(self, rows, weight_fields=None,weight_factor=5.0):
        self.rows = rows
        self.weight_fields = weight_fields or ['Name of Incident', 'description', 'reddit_posts']
        self.weight_factor = weight_factor
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            max_features=5000,
            token_pattern=r'\b[a-zA-Z]{3,}\b'
        )
        
        self.corpus = self._prepare_weighted_corpus()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus)
        self.feature_names = self.vectorizer.get_feature_names_out()
        self.document_vectors = self.tfidf_matrix.toarray()

        self.doc_labels = [
            f"{row.get('Name of Incident', 'Unknown')} ({row.get('Place Name', 'Unknown')})"
            for row in self.rows
        ]
    
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
    
    def get_top_terms(self, n=10):
        top_terms = []
        
        for i, label in enumerate(self.doc_labels):
            tfidf_scores = self.tfidf_matrix[i].toarray().flatten()
            sorted_indices = np.argsort(tfidf_scores)[::-1]
            
            document_terms = {}
            for idx in sorted_indices[:n]:
                if tfidf_scores[idx] > 0:
                    term = self.feature_names[idx]
                    score = tfidf_scores[idx]
                    document_terms[term] = score
            
            top_terms.append({
                'document': label,
                'top_terms': document_terms
            })
        
        return top_terms
    
    def get_document_similarity(self):
        from sklearn.metrics.pairwise import cosine_similarity
        
        similarity_matrix = cosine_similarity(self.tfidf_matrix)
        similarity_df = pd.DataFrame(similarity_matrix, index=self.doc_labels, columns=self.doc_labels)
        
        return similarity_df
    
    def search(self, query, top_n=5):
        print("Query:", query)
        from sklearn.metrics.pairwise import cosine_similarity
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        top_indices = similarities.argsort()[::-1][:top_n]
        
        query_words = set(query.lower().split())
        
        results = []
        for idx in top_indices:
            row = self.rows[idx]
            score = similarities[idx]
            
            if score > 0:
                tfidf_scores = self.tfidf_matrix[idx].toarray().flatten()
                sorted_indices = np.argsort(tfidf_scores)[::-1]
                
                matched_terms = {}
                important_terms = {}
                
                for term_idx in sorted_indices[:20]:  
                    if tfidf_scores[term_idx] > 0:
                        term = self.feature_names[term_idx]
                        score = float(tfidf_scores[term_idx])
                        
                        if term in query_words:
                            matched_terms[term] = score
                        else:
                            important_terms[term] = score                
                important_terms = dict(sorted(important_terms.items(), 
                                            key=lambda x: x[1], 
                                            reverse=True)[:5])
                
                doc_similarities = self.get_document_similarity().iloc[idx].sort_values(ascending=False)
                similar_docs = [
                    {
                        "document": doc_label,
                        "score": float(sim_score),
                        'embedding': self.document_vectors[idx].tolist()
                    }
                    for doc_label, sim_score in doc_similarities[1:3].items()
                ]
                
                themes = [term for term, _ in sorted(important_terms.items(), 
                                                key=lambda x: x[1], 
                                                reverse=True)[:3]]
                
                results.append({
                    'document': self.doc_labels[idx],
                    'score': float(score)*10,
                    'row': row,
                    'themes': themes,
                    'similar_documents': similar_docs,
                    'embedding': self.document_vectors[idx].tolist(),
                    'query_embedding': query_vector.toarray()[0].tolist()
                })
                    
        return results