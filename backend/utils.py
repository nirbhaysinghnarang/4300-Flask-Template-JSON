
import numpy as np
from sklearn.decomposition import PCA

def assign_era(year_str):
    try:
        year_str = str(year_str).strip()
        if 'BC' in year_str.upper():
            year_num = -int(year_str.replace('BC', '').strip())
        elif 'AD' in year_str.upper():
            year_num = int(year_str.replace('AD', '').strip())
        else:
            year_num = int(year_str)
    except:
        return "Unknown"

    if year_num <= -3000:
        return "Prehistoric"
    elif year_num <= -1000:
        return "Bronze Age"
    elif year_num <= 0:
        return "Iron Age / Classical"
    elif year_num <= 500:
        return "Classical Antiquity"
    elif year_num <= 1500:
        return "Medieval"
    elif year_num <= 1800:
        return "Early Modern"
    elif year_num <= 1945:
        return "Colonial / Industrial"
    else:
        return "Contemporary"

def get_data():
    import os
    import json
    os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..", os.curdir))
    current_directory = os.path.dirname(os.path.abspath(__file__))
    json_file_path = os.path.join(current_directory, 'data', 'final_data_with_reddit.json')

    with open(json_file_path, 'r', encoding='utf-8') as f:
        historical_data = json.load(f)
    return historical_data


def get_rows_to_remove():
    import os
    import json
    os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..", os.curdir))
    current_directory = os.path.dirname(os.path.abspath(__file__))
    json_file_path = os.path.join(current_directory, 'data', 'filtered_verified_reddit_posts.json')

    with open(json_file_path, 'r', encoding='utf-8') as f:
        historical_data = json.load(f)
        
    return [datum['Name of Incident'] for datum in historical_data]




import numpy as np
from sklearn.decomposition import PCA

def add_2d_embeddings(filtered_results):
    """
    Add 2D PCA embeddings to search results
    
    Args:
        filtered_results: List of search result dictionaries
        
    Returns:
        The same filtered_results list with 2D embeddings added
    """
    if not filtered_results:
        return filtered_results
    
    all_embeddings = []
    
    for result in filtered_results:
        if 'embedding' in result:
            all_embeddings.append(result['embedding'])
            
        if 'similar_documents' in result:
            for similar_doc in result['similar_documents']:
                if 'embedding' in similar_doc:
                    all_embeddings.append(similar_doc['embedding'])
    
    if not all_embeddings:
        return filtered_results
    
    all_embeddings_np = np.array(all_embeddings)
    
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(all_embeddings_np)
    
    current_idx = 0
    
    for result in filtered_results:
        if 'embedding' in result:
            result['embedding_2d'] = embeddings_2d[current_idx].tolist()
            current_idx += 1
        
        if 'similar_documents' in result:
            for similar_doc in result['similar_documents']:
                if 'embedding' in similar_doc:
                    similar_doc['embedding_2d'] = embeddings_2d[current_idx].tolist()
                    current_idx += 1
    
    if filtered_results and 'query_embedding' in filtered_results[0]:
        query_embeddings = [result['query_embedding'] for result in filtered_results if 'query_embedding' in result]
        
        if query_embeddings:
            query_embeddings_np = np.array(query_embeddings)
            query_embeddings_2d = pca.transform(query_embeddings_np)
            
            query_idx = 0
            for result in filtered_results:
                if 'query_embedding' in result:
                    result['query_embedding_2d'] = query_embeddings_2d[query_idx].tolist()
                    query_idx += 1
    
    return filtered_results