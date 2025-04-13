
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