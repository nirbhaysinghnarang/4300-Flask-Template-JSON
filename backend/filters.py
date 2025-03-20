import pandas as pd
import numpy as np

class Filters:
    def __init__(self, data, min_year: str, max_year: str):
        self.data = data
        self.min_year = self.transform_years(min_year)
        self.max_year = self.transform_years(max_year)
        print(f"min_year: {self.min_year}, max_year: {self.max_year}")
    
    def transform_years(self, year):
        if isinstance(year, str) and "BC" in year:
            year = -1 * int(year.replace("BC", "").strip())
        else:
            try:
                year = int(year)
            except (ValueError, TypeError):
                year = np.nan
        return year
    
    def filter_by_year(self):
        if not isinstance(self.data, list):
            print("Warning: Expected a list of records")
            return self.data
        
        filtered_records = []
        for record in self.data:
            if isinstance(record, dict) and 'row' in record:
                nested_row = record['row']
                if isinstance(nested_row, dict) and 'Year' in nested_row:
                    normalized_year = self.transform_years(nested_row['Year'])
                    if not np.isnan(normalized_year) and self.min_year <= normalized_year <= self.max_year:
                        filtered_records.append(record)
            
        return filtered_records
