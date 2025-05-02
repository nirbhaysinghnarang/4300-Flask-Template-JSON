import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from processor import WeightedTfidfProcessor
from utils import get_data, get_rows_to_remove, assign_era
rows_to_remove = get_rows_to_remove()
historical_data= get_data()

import pandas as pd

historical_df = pd.DataFrame(historical_data)
historical_df['era'] = historical_df['Year'].apply(assign_era)
historical_df["Name of Incident"] = historical_df["Name of Incident"].apply(
    lambda x: str(x).strip().replace("Unknown", "-") if pd.notnull(x) else ""
)

class QueryPreprocessor:
    def __init__(self, tfidf_processor):
        """
        Initialize with a trained WeightedTfidfProcessor to access its vectorizer
        and IDF values.
        
        Args:
            tfidf_processor: A trained WeightedTfidfProcessor instance
        """
        self.tfidf_processor = tfidf_processor
        self.vectorizer = tfidf_processor.vectorizer
        self.idf_values = self.vectorizer.idf_
        self.feature_names = self.vectorizer.get_feature_names_out()
        self.term_idf_map = dict(zip(self.feature_names, self.idf_values))
        self.tf_idf_matrix = tfidf_processor.tfidf_matrix
        print("QueryPreprocessor initialized with IDF values.")

    def create_weighted_cooccurrence_matrix(self):
        import numpy as np
        from scipy.sparse import csr_matrix
        
        tfidf_matrix = self.tfidf_processor.tfidf_matrix                
        cooccurrence_matrix = tfidf_matrix.T.dot(tfidf_matrix)        
        cooccurrence_df = pd.DataFrame(
            cooccurrence_matrix.toarray(),
            index=self.feature_names,
            columns=self.feature_names
        )
        return cooccurrence_df
    
    


    # def expand_query(self, query, threshold=0.95, top_n=5):
    #     """
    #     Expand a query with related terms by combining co-occurrence data across all important query terms,
    #     weighted by the TF-IDF values of the query terms.
        
    #     Args:
    #         query: The original query string
    #         threshold: Minimum TF-IDF score for a term to be considered important
    #         top_n: Number of co-occurring terms to add to the expanded query
            
    #     Returns:
    #         str: Expanded query with additional related terms
    #     """
    #     query_vector = self.vectorizer.transform([query])
    #     cooccurrence_matrix = self.create_weighted_cooccurrence_matrix()
    #     query_terms = [term for term in query.lower().split() 
    #                 if term in self.feature_names]
        
    #     # Identify important terms and their TF-IDF values
    #     important_terms = []
    #     term_weights = {}
        
    #     for term in query_terms:
    #         if term in self.feature_names:
    #             term_idx = list(self.feature_names).index(term)
    #             if term_idx < query_vector.shape[1]:
    #                 term_tfidf = query_vector[0, term_idx]
    #                 if term_tfidf > threshold:
    #                     important_terms.append(term)
    #                     term_weights[term] = term_tfidf
        
    #     if not important_terms:
    #         important_terms = query_terms
    #         # Assign default weights if below threshold
    #         for term in important_terms:
    #             if term in self.feature_names:
    #                 term_idx = list(self.feature_names).index(term)
    #                 if term_idx < query_vector.shape[1]:
    #                     term_weights[term] = query_vector[0, term_idx]
    #                 else:
    #                     term_weights[term] = 0.1  # Default weight
    #             else:
    #                 term_weights[term] = 0.1  # Default weight
        
    #     # Create a weighted combined co-occurrence profile
    #     combined_cooccurrence = pd.Series(0.0, index=self.feature_names)
        
    #     # Weight each term's co-occurrences by its TF-IDF value
    #     for term in important_terms:
    #         if term in cooccurrence_matrix.index:
    #             term_cooccurrences = cooccurrence_matrix.loc[term]
    #             # Multiply co-occurrences by this term's TF-IDF weight
    #             weighted_cooccurrences = term_cooccurrences * term_weights[term]
    #             combined_cooccurrence += weighted_cooccurrences
        
    #     # Filter out original query terms
    #     combined_cooccurrence = combined_cooccurrence[~combined_cooccurrence.index.isin(query_terms)]
        
    #     # Get top co-occurring terms
    #     top_combined_cooccur = combined_cooccurrence.nlargest(top_n)
        
    #     print(f"Top combined co-occurrences for query terms {important_terms}:")
    #     for term, weight in top_combined_cooccur.items():
    #         print(f"  {term}: {weight}")
        
    #     # Expand the query with top terms
    #     expansion_list = top_combined_cooccur.index.tolist()
    #     expanded_query = " ".join(important_terms) + " " + " ".join(expansion_list)
        
    #     print(f"Expanded query: {expanded_query}")
        
    #     return expanded_query
        
    def expand_query(self, query, threshold=0.95, top_n=5, entropy_threshold=0.4):
 
        import numpy as np
        from scipy.stats import entropy
        
        query_vector = self.vectorizer.transform([query])
        cooccurrence_matrix = self.create_weighted_cooccurrence_matrix()
        query_terms = [term for term in query.lower().split() 
                    if term in self.feature_names]
        
        if not query_terms:
            return query
        
        important_terms = []
        term_weights = {}
        
        for term in query_terms:
            if term in self.feature_names:
                term_idx = list(self.feature_names).index(term)
                if term_idx < query_vector.shape[1]:
                    term_tfidf = query_vector[0, term_idx]
                    if term_tfidf > threshold:
                        important_terms.append(term)
                        term_weights[term] = term_tfidf
        
        if not important_terms:
            important_terms = query_terms
            for term in important_terms:
                if term in self.feature_names:
                    term_idx = list(self.feature_names).index(term)
                    if term_idx < query_vector.shape[1]:
                        term_weights[term] = query_vector[0, term_idx]
                    else:
                        term_weights[term] = 0.1 
                else:
                    term_weights[term] = 0.1  
        
        query_specificity = np.mean(list(term_weights.values()))
        adaptive_top_n = max(1, int(top_n * (1 - query_specificity/2)))
        
        print(f"Query specificity: {query_specificity:.4f}, adapting to {adaptive_top_n} expansion terms")
        
        combined_cooccurrence = pd.Series(0.0, index=self.feature_names)
        
        for term in important_terms:
            if term in cooccurrence_matrix.index:
                term_cooccurrences = cooccurrence_matrix.loc[term]
                weighted_cooccurrences = term_cooccurrences * term_weights[term]
                combined_cooccurrence += weighted_cooccurrences
        
        combined_cooccurrence = combined_cooccurrence[~combined_cooccurrence.index.isin(query_terms)]
        
        term_entropy = {}
        total_docs = self.tf_idf_matrix.shape[0]
        
        top_candidates = combined_cooccurrence.nlargest(top_n * 3).index.tolist()
        
        for term in top_candidates:
            if term in self.feature_names:
                term_idx = list(self.feature_names).index(term)
                term_docs = self.tf_idf_matrix[:, term_idx].toarray().flatten()
                term_docs = term_docs / (np.sum(term_docs) or 1)  # Normalize
                
              
                term_entropy[term] = 1.0 - min(1.0, entropy(term_docs) / np.log(total_docs))
        
        entropy_filtered = {term: (combined_cooccurrence[term], ent) 
                            for term, ent in term_entropy.items() 
                            if ent > entropy_threshold}
        
        print(f"Entropy-filtered candidates: {len(entropy_filtered)} of {len(top_candidates)}")
        
        negative_boost = {}
        for term in entropy_filtered:
            neg_cooccur_score = 0
            for other_term in entropy_filtered:
                if term != other_term and term in cooccurrence_matrix.index and other_term in cooccurrence_matrix.columns:
                    neg_cooccur_score += 1 - min(1.0, cooccurrence_matrix.loc[term, other_term] / 
                                            (combined_cooccurrence[term] + 1e-10))
            
            negative_boost[term] = neg_cooccur_score / (len(entropy_filtered) - 1 or 1)
        
        final_scores = {}
        for term in entropy_filtered:
            cooccur_score, entropy_score = entropy_filtered[term]
            neg_score = negative_boost.get(term, 0)
            
            final_scores[term] = (0.5 * cooccur_score) + (0.3 * entropy_score) + (0.2 * neg_score)
        
        final_top_terms = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:adaptive_top_n]
        
        print(f"Top expansion terms with entropy & negative boost:")
        for term, score in final_top_terms:
            entropy_val = entropy_filtered[term][1]
            neg_boost = negative_boost.get(term, 0)
            print(f"  {term}: score={score:.4f}, entropy={entropy_val:.4f}, neg_boost={neg_boost:.4f}")
        
        expansion_list = [term for term, _ in final_top_terms]
        expanded_query = " ".join(important_terms) + " " + " ".join(expansion_list)
        
        print(f"Expanded query: {expanded_query}")
        
        return expanded_query

if  __name__ == '__main__':
    weight_processor_no_social_media = WeightedTfidfProcessor(
    historical_df.to_dict('records'),
    weight_fields=['Name of Incident', 'description'],
    weight_factor=2
)
    
    preprocessor = QueryPreprocessor(weight_processor_no_social_media)
    print(preprocessor.expand_query("fall of communism in europe"))
    
    