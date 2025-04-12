
import json
import os
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
from utils import assign_era, get_data

#Different search methods
from processor import WeightedTfidfProcessor
from lda_processor import LDAProcessor
from glove_processor import GloVEProcessor
from filters import Filters


historical_data= get_data()
historical_df = pd.DataFrame(historical_data)
historical_df['era'] = historical_df['Year'].apply(assign_era)
historical_df["Name of Incident"] = historical_df["Name of Incident"].apply(
    lambda x: str(x).strip().replace("Unknown", "-") if pd.notnull(x) else ""
)
weight_processor_social_media = WeightedTfidfProcessor(
    historical_df.to_dict('records'),
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
    historical_df.to_dict('records'),
    weight_fields=['reddit_posts'],
    weight_factor=20
)

def tfidf_search(query, use_reddit, top_n=5):
    if use_reddit:
        return weight_processor_social_media.search(query, top_n)
    return weight_processor_no_social_media.search(query, top_n)


def glove_search(query, use_reddit, top_n=5):
    if use_reddit:
        return glove_processor_social_media.search(query, top_n)
    return glove_processor_no_social_media.search(query, top_n)


app = Flask(__name__)
CORS(app)


@app.route("/")
def home():
    mapbox_token = os.environ.get('MAPBOX_ACCESS_TOKEN')
    return render_template('base.html', title="World Heritage Explorer", mapbox_token=mapbox_token)


@app.route("/historical-sites")
def historical_search():
    query = request.args.get("query", "")
    min_year = request.args.get("minYear", "2500BC")
    max_year = request.args.get("maxYear", "2012")
    use_reddit = request.args.get("useReddit", False)
    use_glove = request.args.get("useGlove", False)

    print("Query:", query)
    print("Min Year:", min_year)
    print("Max Year:", max_year)
    print("Use Reddit:", use_reddit)
    print("Use GloVe:", use_glove)
    
    if not query:
        return jsonify([])


    if use_glove:
        results = glove_search(query, use_reddit)
    else:
        results = tfidf_search(query, use_reddit)
        
        
    filtered_results = Filters(
        results,
        min_year,
        max_year
    ).filter_by_year()

    return jsonify(filtered_results)





if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8080)
