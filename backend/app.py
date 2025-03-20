import json
import os
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd


from processor import WeightedTfidfProcessor
from filters import Filters

os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

current_directory = os.path.dirname(os.path.abspath(__file__))
json_file_path = os.path.join(current_directory, 'init.json')
csv_file_path = os.path.join(current_directory, 'data', 'final_data.csv')

historical_df = pd.read_csv(csv_file_path)


weight_processor = WeightedTfidfProcessor(
    historical_df.to_dict('records'),
    weight_factor=1
)


def tfidf_search(query, top_n=5):
    return weight_processor.search(query, top_n=top_n)
    
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

    
    
    if not query:
        return jsonify([])
    
    
    results = tfidf_search(query)
    filtered_results = Filters(
        results, 
        min_year, 
        max_year
    ).filter_by_year()
    
    return jsonify(filtered_results)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8080)