from cluster_generator import ClusterGenerator
from utils import get_rows_to_remove, get_data, assign_era
import pandas as pd
#This script will generate and save clusters!
if __name__ == "__main__":
    clusters = [5, 10, 25, 50, 100]
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

    for cluster in clusters:
  
        cluster_generator = ClusterGenerator(historical_df_sm.to_dict('records'),n_clusters=cluster)
        clustered_data = cluster_generator.cluster_data()
        cluster_generator.export_to_json(clustered_data)
