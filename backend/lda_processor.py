import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import gc
import re
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

class LDAProcessor:
    def __init__(self, rows, n_topics=100, max_features=2000, weight_fields=None, weight_factor=5.0):

        self.rows = rows
        self.weight_fields = weight_fields or ['Name of Incident', 'description', 'reddit_posts']
        self.weight_factor = weight_factor
        self.n_topics = n_topics
        self.max_features = max_features
        
        self.corpus = self._prepare_weighted_corpus()
        
        self.vectorizer = CountVectorizer(
            lowercase=True,
            stop_words='english',
            max_features=self.max_features,
            token_pattern=r'\b[a-zA-Z]{3,}\b'
        )
        self.count_matrix = self.vectorizer.fit_transform(self.corpus)
        self.feature_names = np.array(self.vectorizer.get_feature_names_out())
        
        print(f"Running LDA with {n_topics} topics...")
        self.lda_model = LatentDirichletAllocation(
            n_components=n_topics, 
            random_state=42,
            learning_method='online',
            batch_size=128,  # Smaller batch size
            max_iter=10,     # Fewer iterations
            n_jobs=-1        # Use all CPU cores
        )
        self.document_topics = self.lda_model.fit_transform(self.count_matrix)
        
        self.doc_labels = [
            f"{row.get('Name of Incident', 'Unknown')} ({row.get('Place Name', 'Unknown')})"
            for row in self.rows
        ]
        
        # Clean up to save memory
        del self.count_matrix
        gc.collect()
    
    def _clean_text(self, text):
        """Clean text to reduce memory usage"""
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
    
    def get_topics(self, n_words=10):
        """Get the top words for each topic"""
        topics = []
        for topic_idx, topic in enumerate(self.lda_model.components_):
            top_words_idx = topic.argsort()[:-n_words-1:-1]
            top_words = [self.feature_names[i] for i in top_words_idx]
            top_words_weights = [topic[i] for i in top_words_idx]
            
            topics.append({
                'topic_id': topic_idx,
                'words': dict(zip(top_words, top_words_weights))
            })
        
        return topics
    
    def get_document_topics(self, doc_idx=None):
        if doc_idx is not None:
            return {
                'document': self.doc_labels[doc_idx],
                'topics': {i: score for i, score in enumerate(self.document_topics[doc_idx])}
            }
        else:
            results = []
            for i, label in enumerate(self.doc_labels):
                results.append({
                    'document': label,
                    'topics': {j: score for j, score in enumerate(self.document_topics[i])}
                })
            return results
    
    def get_document_similarity(self):
        """Calculate document similarity matrix based on topic distributions"""
        print("Calculating similarity matrix...")
        similarity_matrix = cosine_similarity(self.document_topics)
        similarity_df = pd.DataFrame(
            similarity_matrix, 
            index=self.doc_labels, 
            columns=self.doc_labels
        )
        return similarity_df
    
    def search(self, query, top_n=5):
        """Search for similar documents based on topic distribution"""
        clean_query = self._clean_text(query)
        query_vector = self.vectorizer.transform([clean_query])        
        query_topics = self.lda_model.transform(query_vector)[0]        
        similarities = cosine_similarity([query_topics], self.document_topics).flatten()        
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
    
    def __del__(self):
        """Clean up memory when object is deleted"""
        del self.lda_model
        del self.document_topics
        del self.vectorizer
        gc.collect()
