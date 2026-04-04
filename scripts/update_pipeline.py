import os
import sys
import pandas as pd
import kaggle
import logging

# Append root directory to path to allow importing your existing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train import run_training_pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
KAGGLE_DATASET = 'asaniczka/tmdb-movies-dataset-2023-930k-movies'
TARGET_TMDB_FILE = os.path.join(DATA_DIR, "tmdb_movies_lite.csv") # Must match your load_tmdb_dataset() path
MIN_VOTE_COUNT = 500

def run_etl():
    """Download and downsample the TMDB dataset."""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    logger.info("Authenticating with Kaggle API...")
    kaggle.api.authenticate()
    
    logger.info("Downloading raw TMDB dataset...")
    kaggle.api.dataset_download_files(KAGGLE_DATASET, path=DATA_DIR, unzip=True)
    
    # Locate the extracted file (Kaggle typically extracts this as TMDB_movie_dataset_v11.csv)
    raw_csv_path = os.path.join(DATA_DIR, "TMDB_movie_dataset_v11.csv")
    
    logger.info("Loading and filtering dataset to satisfy memory constraints...")
    raw_df = pd.read_csv(raw_csv_path)
    
    # Enforce downsampling to prevent Render OOM and GitHub file limit errors
    lite_df = raw_df[raw_df['vote_count'] >= MIN_VOTE_COUNT].copy()
    
    # Drop heavyweight columns strictly unnecessary for CF or HMM evaluation
    drop_cols = ['overview', 'tagline', 'homepage', 'production_companies', 'backdrop_path']
    lite_df.drop(columns=[c for c in drop_cols if c in lite_df.columns], inplace=True)
    
    logger.info(f"Writing optimized dataset ({lite_df.shape[0]} rows) to {TARGET_TMDB_FILE}...")
    lite_df.to_csv(TARGET_TMDB_FILE, index=False)
    
    # Cleanup raw 150MB+ file
    os.remove(raw_csv_path)
    logger.info("ETL Phase Complete.")

if __name__ == "__main__":
    try:
        # Phase 1: Prepare the data
        run_etl()
        
        # Phase 2: Execute your proprietary training pipeline
        logger.info("Initiating core training sequence...")
        artifacts = run_training_pipeline()
        
        logger.info("CI/CD Pipeline executed successfully.")
        
    except Exception as e:
        logger.error("FATAL: Pipeline execution failed.", exc_info=True)
        sys.exit(1)
