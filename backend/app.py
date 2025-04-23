
import json
import os
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
from utils import assign_era, get_data, get_rows_to_remove, add_2d_embeddings

#Different search methods
from processor import WeightedTfidfProcessor
from glove_processor import GloVEProcessor
from svd_processor import SVDTextProcessor
from filters import Filters

rows_to_remove = get_rows_to_remove()
historical_data= get_data()



historical_df = pd.DataFrame(historical_data)
historical_df['era'] = historical_df['Year'].apply(assign_era)
historical_df["Name of Incident"] = historical_df["Name of Incident"].apply(
    lambda x: str(x).strip().replace("Unknown", "-") if pd.notnull(x) else ""
)


historical_df_sm = historical_df[~historical_df['Name of Incident'].isin(rows_to_remove)]


print("Dataset statistics")
print("Total rows:", len(historical_df))
print("Rows to remove:", len(rows_to_remove))
print("Rows remaining:", len(historical_df_sm))



weight_processor_social_media = WeightedTfidfProcessor(
    historical_df_sm.to_dict('records'),
    weight_fields=['reddit_posts'],
    weight_factor=10
)

weight_processor_no_social_media = WeightedTfidfProcessor(
    historical_df.to_dict('records'),
    weight_fields=['Name of Incident', 'description'],
    weight_factor=2
)

glove_processor_no_social_media = GloVEProcessor(
    historical_df.to_dict('records'),
    weight_fields=['Name of Incident', 'description'],
    weight_factor=2
)

glove_processor_social_media = GloVEProcessor(
    historical_df_sm.to_dict('records'),
    weight_fields=['reddit_posts'],
    weight_factor=20
)


svd_processor_no_social_media = SVDTextProcessor(
    historical_df.to_dict('records'),
    weight_fields=['Name of Incident', 'description'],
    weight_factor=10
)


svd_processor_social_media = SVDTextProcessor(
    historical_df_sm.to_dict('records'),
    weight_fields=['reddit_posts'],
    weight_factor=10
)

def tfidf_search(query, use_reddit, top_n=5):
    if use_reddit:
        return weight_processor_social_media.search(query, top_n)
    return weight_processor_no_social_media.search(query, top_n)


def glove_search(query, use_reddit, top_n=5):
    if use_reddit:
        return glove_processor_social_media.search(query, top_n)
    return glove_processor_no_social_media.search(query, top_n)


def svd_search(query, use_reddit, top_n=5):
    if use_reddit:
        return svd_processor_social_media.search(query, top_n)
    return svd_processor_no_social_media.search(query, top_n)

app = Flask(__name__)
CORS(app)


@app.route("/")
def home():
    mapbox_token = os.environ.get('MAPBOX_ACCESS_TOKEN')
    return render_template('base.html', title="World Heritage Explorer", mapbox_token=mapbox_token)

@app.route("/svd")
def svd():
    return render_template('svd.html', title="SVD Concepts Explorer")



@app.route("/svd/query")
def get_concepts():
    return svd_processor_no_social_media.get_term_concept_matrix()


@app.route("/historical-sites")
def historical_search():
    query = request.args.get("query", "")
    min_year = request.args.get("minYear", "2500BC")
    max_year = request.args.get("maxYear", "2025")
    use_reddit = request.args.get("useReddit", False)
    embedding_method = request.args.get("embeddingMethod","TF")
    
    print(request.args)
    if embedding_method not in [
        "TF",
        "GLOVE",
        "SVD"
    ]:
        embedding_method = "TF"
        
    
    print("Query:", query)
    print("Min Year:", min_year)
    print("Max Year:", max_year)
    print("Use Reddit:", use_reddit)
    print("Embedding Method:", embedding_method)
    
    if not query:
        return jsonify([])

    if embedding_method == "TF":
        results = tfidf_search(query, use_reddit)
    elif embedding_method == "GLOVE":
        results = glove_search(query, use_reddit)
    elif embedding_method == "SVD":
        results = svd_search(query, use_reddit)
    
    filtered_results = Filters(
        results,
        min_year,
        max_year
    ).filter_by_year()
    print([res['row']['Name of Incident'] for res in filtered_results])
    # filtered_results = add_2d_embeddings(filtered_results)
    # print(filtered_results[0])        
    return jsonify(filtered_results)


@app.route("/clusters")
def render_clusters():
    return render_template('cluster-viz.html', title="Clusters", clustered_data=None)


@app.route("/get-clusters")
def get_clusters():
    with open("clustered_data_100.json", "r") as f:
        data = json.load(f)
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8081)
