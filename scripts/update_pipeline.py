import os
import pandas as pd
import joblib
import kaggle
# Import your training functions from your existing modules
from model_training import train_hybrid_model 

def run_pipeline():
    print("Authenticating and downloading TMDB dataset from Kaggle...")
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        'asaniczka/tmdb-movies-dataset-2023-930k-movies', 
        path='data/', 
        unzip=True
    )
    
    # 1. Load and downsample
    print("Loading and filtering dataset...")
    raw_df = pd.read_csv("data/TMDB_movie_dataset_v11.csv") # Adjust filename to Kaggle's exact output
    
    # Aggressive filtering to keep the matrix small (e.g., top 15,000 most voted movies)
    lite_df = raw_df[raw_df['vote_count'] > 500].copy()
    
    # Drop unneeded text columns to save space
    columns_to_drop = ['overview', 'tagline', 'homepage', 'poster_path']
    lite_df.drop(columns=[c for c in columns_to_drop if c in lite_df.columns], inplace=True)
    
    # Save the lightweight dataset
    lite_df.to_csv("data/tmdb_movies_lite.csv", index=False)
    print(f"Filtered dataset saved. Final shape: {lite_df.shape}")

    # 2. Retrain models based strictly on the lightweight dataset
    print("Regenerating ML artifacts...")
    tmdb_latent, mlb, ridge = train_hybrid_model(lite_df)
    
    # 3. Serialize and overwrite old models
    joblib.dump(tmdb_latent, "models/tmdb_latent.pkl")
    joblib.dump(mlb, "models/mlb.pkl")
    joblib.dump(ridge, "models/ridge.pkl")
    print("Pipeline complete. Artifacts updated.")

if __name__ == "__main__":
    run_pipeline()