import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
import sys
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')

class PretrainedWord2VecProcessor:
    def __init__(self, rows, weight_fields=None, pretrained_model_path=None):
        self.rows = rows
        self.weight_fields = weight_fields or ['Name of Incident', 'description']
        
        if pretrained_model_path is None:
            import gensim.downloader as api
            self.model = api.load('glove-wiki-gigaword-50')  # Much smaller model (25-dim instead of 300-dim)

        else:
            self.model = KeyedVectors.load_word2vec_format(pretrained_model_path, binary=True)
        
        self.documents = self._prepare_documents()
        self.doc_labels = [
            f"{row.get('Name of Incident', 'Unknown')} ({row.get('Place Name', 'Unknown')})"
            for row in self.rows
        ]
        
        self.doc_vectors = self._create_document_vectors()
    
    def _prepare_documents(self):
        documents = []
        
        for row in self.rows:
            doc_parts = []
            
            for key, value in row.items():
                if value and isinstance(value, (str, int, float)):
                    if isinstance(value, str):
                        doc_parts.extend(word_tokenize(str(value).lower()))
                    else:
                        doc_parts.extend(word_tokenize(str(value)))
            
            for field in self.weight_fields:
                if field in row and row[field]:
                    doc_parts.extend(word_tokenize(str(row[field]).lower()))
            
            documents.append(doc_parts)
        
        return documents
    
    def _create_document_vectors(self):
        doc_vectors = []
        
        for doc in self.documents:
            valid_words = [word for word in doc if word in self.model.key_to_index]
            
            if not valid_words:
                doc_vector = np.zeros(self.model.vector_size)
            else:
                doc_vector = np.mean([self.model[word] for word in valid_words], axis=0)
            
            doc_vectors.append(doc_vector)
        
        return doc_vectors
    
    def get_top_terms(self, n=10):
        top_terms = []
        
        for i, label in enumerate(self.doc_labels):
            doc_vector = self.doc_vectors[i]
            
            if np.all(doc_vector == 0):
                top_terms.append({
                    'document': label,
                    'top_terms': {}
                })
                continue
                
            similar_words = self.model.similar_by_vector(doc_vector, topn=n)
            
            document_terms = {}
            for word, score in similar_words:
                document_terms[word] = score
            
            top_terms.append({
                'document': label,
                'top_terms': document_terms
            })
        
        return top_terms
    
    def get_document_similarity(self):
        similarity_matrix = np.zeros((len(self.documents), len(self.documents)))
        
        for i in range(len(self.documents)):
            for j in range(len(self.documents)):
                vec_i = self.doc_vectors[i]
                vec_j = self.doc_vectors[j]
                
                if np.all(vec_i == 0) or np.all(vec_j == 0):
                    similarity = 0
                else:
                    similarity = np.dot(vec_i, vec_j) / (np.linalg.norm(vec_i) * np.linalg.norm(vec_j))
                
                similarity_matrix[i, j] = similarity
        
        similarity_df = pd.DataFrame(similarity_matrix, index=self.doc_labels, columns=self.doc_labels)
        
        return similarity_df
    
    def search(self, query, top_n=5):
        tokenized_query = word_tokenize(query.lower())
        
        valid_words = [word for word in tokenized_query if word in self.model.key_to_index]
        if not valid_words:
            return []
        
        query_vector = np.mean([self.model[word] for word in valid_words], axis=0)
        
        most_similar_docs = []
        for i in range(len(self.documents)):
            doc_vector = self.doc_vectors[i]
            
            if np.all(doc_vector == 0):
                similarity = 0
            else:
                similarity = np.dot(query_vector, doc_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(doc_vector))
            
            most_similar_docs.append((i, similarity))
        
        most_similar_docs.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, score in most_similar_docs[:top_n]:
            row = self.rows[idx]
            if score > 0:
                results.append({
                    'document': self.doc_labels[idx],
                    'score': float(score),
                    'row': row
                })
        
        return results


os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the JSON file relative to the current script
json_file_path = os.path.join(current_directory, 'init.json')

# Assuming your JSON data is stored in a file named 'init.json'
with open(json_file_path, 'r') as file:
    data = json.load(file)
    episodes_df = pd.DataFrame(data['episodes'])
    reviews_df = pd.DataFrame(data['reviews'])

pretrained_word2vec_processor = PretrainedWord2VecProcessor(
    historical_df.to_dict('records'),
    weight_fields=['Name of Incident', 'description']
)

def word2vec_search(query, top_n=5):
    return pretrained_word2vec_processor.search(query, top_n=top_n)
    
app = Flask(__name__)
CORS(app)

# Sample search using json with pandas
def json_search(query):
    matches = []
    merged_df = pd.merge(episodes_df, reviews_df, left_on='id', right_on='id', how='inner')
    matches = merged_df[merged_df['title'].str.lower().str.contains(query.lower())]
    matches_filtered = matches[['title', 'descr', 'imdb_rating']]
    matches_filtered_json = matches_filtered.to_json(orient='records')
    return matches_filtered_json

@app.route("/")
def home():
    mapbox_token = os.environ.get('MAPBOX_ACCESS_TOKEN')
    return render_template('base.html', title="World Heritage Explorer", mapbox_token=mapbox_token)

@app.route("/historical-sites")
def historical_search():
    query = request.args.get("query", "")
    if not query:
        return jsonify([])
    results = word2vec_search(query)
    print(results)
    print("SUCCESS")
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8080)
