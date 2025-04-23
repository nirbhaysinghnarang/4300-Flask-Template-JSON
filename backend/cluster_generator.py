from sklearn.manifold import TSNE
import numpy as np
from sklearn.cluster import KMeans
import json
from glove_processor import GloVEProcessor
class ClusterGenerator:
    
    def __init__(self, rows, n_clusters=50):
        self.rows = rows
        self.glove_processor = GloVEProcessor(rows)
        self.rows_embedded = self.glove_processor.dump_rows_with_embeddings()
        self.n_clusters = n_clusters
    
    def reduce_to_3d(self, perplexity=30, n_iter=1000, random_state=42):
    # Extract embeddings from DataFrame
        embeddings = np.array(self.rows_embedded["embedding"].tolist())
        
        # Apply t-SNE dimensionality reduction
        tsne = TSNE(n_components=3, perplexity=perplexity, 
                    n_iter=n_iter, random_state=random_state)
        embeddings_3d = tsne.fit_transform(embeddings)
        
        # Add 3D embeddings as new columns in the DataFrame
        self.rows_embedded["x"] = embeddings_3d[:, 0]
        self.rows_embedded["y"] = embeddings_3d[:, 1]
        self.rows_embedded["z"] = embeddings_3d[:, 2]
        
        return self.rows_embedded
    
    
    def cluster_data(self, dim_reduction=True, random_state=42):
        if dim_reduction:
            data_with_embeddings = self.reduce_to_3d(random_state=random_state)
            embeddings = np.array(data_with_embeddings[["x", "y", "z"]])
        else:
            data_with_embeddings = self.rows_embedded
            embeddings = np.array(data_with_embeddings["embedding"].tolist())
        
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=random_state)
        cluster_labels = kmeans.fit_predict(embeddings)        
        data_with_embeddings["cluster"] = cluster_labels.astype(int)
        
        return data_with_embeddings
    
    def export_to_json(self, df, filename=None, orient="records", indent=2):
        if not filename:
            filename = f"clustered_data_{self.n_clusters}.json"
        # Convert DataFrame to JSON and write to file
        with open(filename, 'w') as f:
            json.dump(df.to_dict(orient=orient), f, indent=indent)
        
        return filename