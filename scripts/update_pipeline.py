import os
import sys
import pandas as pd
import kaggle
import logging

# Append root directory to sys.path to allow importing native modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import TMDB_MOVIES
from train import main as run_training  # <--- THIS IS THE FIX

# Configure strict logging for CI/CD tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
KAGGLE_DATASET = 'asaniczka/tmdb-movies-dataset-2023-930k-movies'
DATA_DIR = os.path.dirname(os.path.abspath(TMDB_MOVIES))
MIN_VOTE_COUNT = 500  

def execute_size_reduction_etl():
    """Download the raw TMDB dataset and strictly reduce its file size."""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    logger.info("Authenticating with Kaggle API...")
    kaggle.api.authenticate()
    
    logger.info("Downloading massive raw TMDB dataset...")
    kaggle.api.dataset_download_files(KAGGLE_DATASET, path=DATA_DIR, unzip=True)
    
    # Kaggle extracts this specific dataset as 'TMDB_movie_dataset_v11.csv'
    raw_csv_path = os.path.join(DATA_DIR, "TMDB_movie_dataset_v11.csv")
    
    logger.info("Loading raw dataset into memory for aggressive downsampling...")
    raw_df = pd.read_csv(raw_csv_path, low_memory=False)
    
    # 1. Row Reduction: Keep only statistically significant movies
    lite_df = raw_df[raw_df['vote_count'] >= MIN_VOTE_COUNT].copy()
    
    # 2. Column Reduction: Drop heavyweight text arrays unused by CF/HMM models
    columns_to_drop = [
        'overview', 'tagline', 'homepage', 'production_companies', 
        'spoken_languages', 'backdrop_path', 'poster_path', 'credits'
    ]
    lite_df.drop(columns=[c for c in columns_to_drop if c in lite_df.columns], inplace=True)
    
    logger.info(f"Writing optimized dataset ({lite_df.shape[0]} rows) to {TMDB_MOVIES}...")
    lite_df.to_csv(TMDB_MOVIES, index=False)
    
    # 3. Purge the raw file to prevent GitHub rejection and free up runner disk space
    os.remove(raw_csv_path)
    logger.info("ETL Size Reduction Phase Complete.")

if __name__ == "__main__":
    try:
        # Stage 1: File Size Optimization
        execute_size_reduction_etl()
        
        # Stage 2: Matrix Computation
        logger.info("Initiating core hybrid training sequence...")
        run_training()  # <--- THIS IS THE SECOND FIX
        
        logger.info("CI/CD Pipeline executed successfully.")
        
    except Exception as e:
        logger.error("FATAL: Pipeline execution failed.", exc_info=True)
        sys.exit(1)
