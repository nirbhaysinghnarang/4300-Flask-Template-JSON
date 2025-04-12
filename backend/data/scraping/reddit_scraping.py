import requests
import csv
import json
import time
import os
from urllib.parse import quote
from typing import List, Dict, Any


def extract_reddit_links_from_response(response_data: Dict) -> List[Dict[str, Any]]:
    reddit_posts = []
    
    if "data" in response_data and "children" in response_data["data"]:
        for post in response_data["data"]["children"]:
            if "data" in post:
                post_data = post["data"]                
                post_info = {
                    "title": post_data.get("title", ""),
                    "url": f"https://www.reddit.com{post_data.get('permalink', '')}" if "permalink" in post_data else post_data.get("url", ""),
                    "score": post_data.get("score", 0),
                    "num_comments": post_data.get("num_comments", 0),
                    "subreddit": post_data.get("subreddit_name_prefixed", ""),
                    "author": post_data.get("author", ""),
                    "created_utc": post_data.get("created_utc", 0)
                }
                
                reddit_posts.append(post_info)
    
    return reddit_posts


def find_reddit_links(event_name: str, event_year: str, event_country: str, max_retries=3) -> List[Dict[str, Any]]:
    query = f"{event_name.replace('Unknown', '')},{event_year}, {event_country}"
    encoded_query = quote(query)
    
    url = f"https://www.reddit.com/search.json?q={encoded_query}&limit=5"
    
    headers = {
        "User-Agent": "Mozilla/5.0 Historical-Event-Search/1.0"
    }
    
    for retry in range(max_retries):
        try:
            response = requests.get(url, headers=headers)
            
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 120))  
                print(f"Rate limited by Reddit (429). Waiting for {retry_after} seconds before retry...")
                time.sleep(retry_after)
                continue  
                
            response.raise_for_status() 
            data = response.json()
            
            return extract_reddit_links_from_response(data)
        
        except requests.exceptions.HTTPError as e:
            if retry < max_retries - 1:  # Don't sleep on the last iteration
                wait_time = 120  # 2 minutes
                print(f"Search failed for {event_name}: {str(e)}")
                print(f"Waiting {wait_time} seconds before retry. Attempt {retry+1}/{max_retries}")
                time.sleep(wait_time)
            else:
                print(f"All retries failed for {event_name}: {str(e)}")
                return []
        
        except Exception as e:
            print(f"Search failed for {event_name}: {str(e)}")
            return []


def process_historical_events_to_json(start_index=None):
    input_file = "../final_data.csv"
    output_file = "../final_data_with_reddit.json"
    checkpoint_file = "../checkpoint.json"
    
    if start_index is not None:
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    result = json.load(f)
                    if start_index < len(result):
                        print(f"Warning: Starting from index {start_index}, but already have {len(result)} records.")
                        print("Do you want to truncate existing results or append? Truncating...")
                        result = result[:start_index]
                    processed_count = start_index
                    print(f"Using existing data and starting from manual index: {start_index}")
            except Exception as e:
                print(f"Error loading existing output file: {str(e)}")
                result = []
                processed_count = start_index
                print(f"Starting from manually specified index: {start_index}")
        else:
            result = []
            processed_count = start_index
            print(f"Starting from manually specified index: {start_index}")
    elif os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                result = json.load(f)
                processed_count = len(result)
                print(f"Checkpoint found! Resuming from record {processed_count}")
        except Exception as e:
            print(f"Error loading checkpoint: {str(e)}")
            result = []
            processed_count = 0
    else:
        result = []
        processed_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    rows_to_process = rows[processed_count:]
    
    for i, row in enumerate(rows_to_process, start=processed_count):
        event_name = row["Name of Incident"]
        event_year = row["Year"]
        event_country = row["Country"]
        
        print(f"[{i+1}/{len(rows)}] Searching: {event_name} ({event_year}) {event_country}")
        
        reddit_posts = find_reddit_links(event_name, event_year, event_country)
        
        row_data = dict(row)
        row_data["reddit_posts"] = reddit_posts
        
        result.append(row_data)
        
        print(f"Found {len(reddit_posts)} Reddit posts")
        
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        if i < len(rows) - 1:
            time.sleep(2)
    
    print(f"Processing complete. Output saved to {output_file}")
    
    try:
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
            print(f"Checkpoint file {checkpoint_file} removed")
    except Exception as e:
        print(f"Warning: Could not remove checkpoint file: {str(e)}")


# LOOK ON OR AROUND INDEX 461  # Set to 0 to start from beginning, or any number to start from that index

START_INDEX = 1116  
def resume_from_checkpoint():
    """Function to manually resume processing from checkpoint"""
    process_historical_events_to_json()
    
def start_from_index(index):
    """Function to manually start processing from a specific index"""
    process_historical_events_to_json(start_index=index)


if __name__ == "__main__":
    process_historical_events_to_json()
    