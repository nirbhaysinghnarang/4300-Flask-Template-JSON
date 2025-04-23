
from pydoc import doc
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors
import gensim.downloader as api
from collections import defaultdict
import gc
import re
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import json
class GloVEProcessor:
    def __init__(self, rows, vector_dim=50, weight_fields=None, weight_factor=5.0):
        self.rows = rows
        self.weight_fields = weight_fields or ['Name of Incident', 'description', 'reddit_posts']
        self.weight_factor = weight_factor
        self.vector_dim = vector_dim
        
        print(f"Loading GloVe model with {vector_dim} dimensions...")
        self.word_vectors = self._load_glove_model()
        
        print("Preparing corpus...")
        self.corpus = self._prepare_weighted_corpus()
        
        print("Creating document vectors...")
        self.document_vectors = self._get_document_vectors()
        
        self.doc_labels = [
            f"{row.get('Name of Incident', 'Unknown')} ({row.get('Place Name', 'Unknown')})"
            for row in self.rows
        ]
        
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            max_features=5000,
            token_pattern=r'\b[a-zA-Z]{3,}\b'
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus)
    
    def _load_glove_model(self):
        try:
            return api.load(f"glove-wiki-gigaword-{self.vector_dim}")
        except Exception as e:
            print(f"Error loading pre-trained model: {e}")
            print("Falling back to loading from file if available...")    
            raise Exception("Could not load GloVe model. Please download it manually or ensure gensim-data is installed.")
    
    def _clean_text(self, text):
        if not isinstance(text, str):
            return ""
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _prepare_weighted_corpus(self):
        """Prepare the corpus with weighted important fields"""
        corpus = []
        for row in tqdm(self.rows, desc="Processing documents"):
            document_parts = []
            for key, value in row.items():
                if value and isinstance(value, (str, int, float)):
                    document_parts.append(self._clean_text(str(value)))            
            for field in self.weight_fields:
                if field in row and row[field]:
                    for _ in range(int(self.weight_factor) - 1):
                        document_parts.append(self._clean_text(str(row[field])))
            
            corpus.append(" ".join(document_parts))
            
        return corpus
    
    def _get_document_vectors(self):
        """Process documents in batches to save memory"""
        document_vectors = []
        for i, doc in enumerate(tqdm(self.corpus, desc="Vectorizing documents")):
            words = [w for w in doc.split() if w in self.word_vectors]
            if words:
                vec_sum = np.zeros(self.vector_dim)
                for word in words:
                    vec_sum += self.word_vectors[word]
                document_vectors.append(vec_sum / len(words))
            else:
                document_vectors.append(np.zeros(self.vector_dim))            
            if i % 100 == 0:
                gc.collect()
        
        return np.array(document_vectors)
    
    def get_document_similarity(self):
        """Calculate document similarity matrix"""
        print("Calculating similarity matrix...")
        similarity_matrix = cosine_similarity(self.document_vectors)
        similarity_df = pd.DataFrame(similarity_matrix, index=self.doc_labels, columns=self.doc_labels)
        return similarity_df
    
    def search(self, query, top_n=5):
        clean_query = self._clean_text(query)
        words = [w for w in clean_query.split() if w in self.word_vectors]
        if not words:
            return []
        
        query_doc = [clean_query]
        
        query_tfidf = self.vectorizer.transform(query_doc).toarray()[0]
        
        word_indices = {}
        for w in words:
            for i, feature in enumerate(self.vectorizer.get_feature_names_out()):
                if w == feature:
                    word_indices[w] = i
                    break
        
        weighted_vectors = []
        for word in words:
            if word in word_indices:
                idx = word_indices[word]
                weight = query_tfidf[idx]
                
                if weight >= 0:
                    print(f"Word: {word}, Weight: {weight}")
                    weighted_vectors.append(self.word_vectors[word] * weight)
        
        if weighted_vectors:
            query_vector = np.sum(weighted_vectors, axis=0)
            norm = np.linalg.norm(query_vector)
            if norm > 0:
                query_vector = query_vector / norm
            query_vector = query_vector.reshape(1, -1)
        else:
            query_vector = np.mean([self.word_vectors[w] for w in words], axis=0).reshape(1, -1)
        
        similarities = cosine_similarity(query_vector, self.document_vectors).flatten()
        top_indices = similarities.argsort()[::-1][:top_n]
        
        
        
        results = []
        for idx in top_indices:
            row = self.rows[idx]
            score = similarities[idx]
                
            doc_similarities = self.get_document_similarity().iloc[idx].sort_values(ascending=False)
            similar_docs = [
                {
                    "document": doc_label,
                    "score": float(sim_score),
                    'embedding': self.document_vectors[idx].tolist()
                }
                    for doc_label, sim_score in doc_similarities[1:3].items()
            ]
                
            if score > 0:
                results.append({
                    'document': self.doc_labels[idx],
                    'score': float(score),
                    'row': row,
                    'similar_documents': similar_docs,
                    'themes': [self.get_theme(idx)],
                    'embedding': self.document_vectors[idx].tolist(),
                    'query_embedding': query_vector.tolist()
                })
        
        return results
    
    
    def get_theme(self, doc_idx):
        print(self.rows[doc_idx])
        idx_name = "Name of Incident"
        with open('clustered_data_100.json', 'r') as f:
            clustered_data = json.load(f)
            
            element = [x for x in clustered_data if x[idx_name] == self.rows[doc_idx][idx_name]]
            if element and len(element) > 0:
                return element[0]['cluster_name']
            return "This document was ignored in clustering."
    
    def get_closest_documents(self, doc_idx, top_n=5):
        """Find documents most similar to the given document"""
        similarities = cosine_similarity([self.document_vectors[doc_idx]], self.document_vectors).flatten()
        
        similarities[doc_idx] = 0
        
        top_indices = similarities.argsort()[::-1][:top_n]
        
        results = []
        for idx in top_indices:
            row = self.rows[idx]
            score = similarities[idx]
            
            if score > 0:
                results.append({
                    'document': self.doc_labels[idx],  
                    'score': score,
                    'row': row,
                    'embedding': self.document_vectors[idx].tolist()
                })
        
        return results
    
    def __del__(self):
        """Clean up memory when object is deleted"""
        del self.word_vectors
        del self.document_vectors
        gc.collect()


    def dump_rows_with_embeddings(self):
        """
        Dump rows with their corresponding document embeddings as a DataFrame.
        
        Returns:
            pd.DataFrame: A DataFrame containing original rows and their embeddings
        """
        rows_with_embeddings = [row.copy() for row in self.rows]            
        for i, row in enumerate(rows_with_embeddings):
            row['embedding'] = self.document_vectors[i].tolist()
        return pd.DataFrame(rows_with_embeddings)