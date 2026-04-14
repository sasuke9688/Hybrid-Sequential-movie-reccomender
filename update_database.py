import os
import time
import pandas as pd
from supabase import create_client, Client

# 1. Environment Configuration
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Supabase credentials missing from environment.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def run_pipeline():
    print("Initiating Kaggle download...")
    # The kaggle library automatically authenticates using KAGGLE_USERNAME and KAGGLE_KEY
    import kaggle
    
    # Replace 'user/dataset-slug' with the exact TMDB dataset slug from Kaggle's URL
    kaggle_dataset = "https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies" # (Or whatever yours actually is)
    kaggle.api.dataset_download_files(kaggle_dataset, path=".", unzip=True)
    
    # Update this filename to match the extracted CSV
    csv_filename = "tmdb_movies.csv" 
    print(f"Loading {csv_filename} into Pandas...")
    
    # 2. Extract and Transform
    df = pd.read_csv(csv_filename)
    
    # Ensure NaN values are converted to None for PostgreSQL compatibility
    df = df.where(pd.notnull(df), None)
    
    # Ensure the pandas index is explicitly saved as a column
    if "pandas_index" not in df.columns:
        df["pandas_index"] = df.index

    # 3. Load (Batched Upsert)
    records = df.to_dict(orient="records")
    batch_size = 1000
    total_records = len(records)
    
    print(f"Initiating Supabase upload. Total records: {total_records}")
    
    for i in range(0, total_records, batch_size):
        batch = records[i : i + batch_size]
        try:
            # upsert relies on the primary key to update existing rows or insert new ones
            supabase.table("tmdb_movies").upsert(batch).execute()
            print(f"Successfully processed batch {i} to {i + len(batch)}")
            time.sleep(0.5) # Rate limiting buffer
        except Exception as e:
            print(f"Failed to process batch {i}: {e}")

if __name__ == "__main__":
    run_pipeline()
