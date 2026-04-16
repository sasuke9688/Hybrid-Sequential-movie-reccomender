import os
import sys
import logging
import pandas as pd
import zipfile

# Configure Logging exactly as your GitHub Actions expects
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
KAGGLE_DATASET = "asaniczka/tmdb-movies-dataset-2023-930k-movies"
DATA_DIR = "data"
OUTPUT_CSV_PATH = os.path.join(DATA_DIR, "tmdb_movies_lite.csv")

def download_kaggle_data():
    logger.info("Authenticating with Kaggle API...")
    try:
        import kaggle
        logger.info("Downloading massive raw TMDB dataset...")
        # Downloads and automatically unzips the dataset
        kaggle.api.dataset_download_files(KAGGLE_DATASET, path='.', unzip=True)
        
        # Find the extracted CSV file
        csv_files = [f for f in os.listdir('.') if f.endswith('.csv') and 'TMDB' in f.upper()]
        if not csv_files:
            csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
            
        if not csv_files:
            raise FileNotFoundError("Could not find extracted Kaggle CSV file.")
        
        return csv_files[0]
        
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        sys.exit(1)

def process_raw_dataset(raw_csv_path, output_csv_path):
    logger.info("Loading raw dataset into memory for stratified linguistic extraction...")
    
    # Load the massive raw dataset
    df_raw = pd.read_csv(raw_csv_path, low_memory=False)
    
    # 1. Apply a baseline quality threshold to eliminate noise (zero-vote entries)
    if 'vote_count' in df_raw.columns:
        df_filtered = df_raw[df_raw['vote_count'] >= 50].copy()
    else:
        df_filtered = df_raw.copy()
        
    # 2. Sort by popularity globally to ensure we extract the highest-rated films in each category
    if 'popularity' in df_filtered.columns:
        df_filtered = df_filtered.sort_values(by='popularity', ascending=False)
        
    # 3. Stratified Extraction: 5000 English
    df_en = df_filtered[df_filtered['original_language'] == 'en'].head(5000)
    
    # 4. Stratified Extraction: 1000 Japanese
    df_ja = df_filtered[df_filtered['original_language'] == 'ja'].head(1000)
    
    # 5. Stratified Extraction: 1500 Regional Indian (Hindi, Telugu, Tamil, Malayalam)
    indian_langs = ['hi', 'te', 'ta', 'ml']
    df_in = df_filtered[df_filtered['original_language'].isin(indian_langs)].head(1500)
    
    # 6. Concatenate into the final 7,500-row dataset
    df_final = pd.concat([df_en, df_ja, df_in], ignore_index=True)
    
    # 7. Shuffle the dataset to ensure homogeneous distribution during ML training batches
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Ensure data directory exists
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    
    # 8. Write the optimized dataset to the target pipeline location
    logger.info(f"Writing stratified dataset ({len(df_final)} rows) to {output_csv_path}...")
    df_final.to_csv(output_csv_path, index=False)
    logger.info("ETL Size Reduction Phase Complete.")

def run_training():
    logger.info("Initiating core hybrid training sequence...")
    try:
        # Ensure the root directory is in the Python path so it can find model_training.py
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from model_training import run_training_pipeline
        
        # Trigger the PyTorch neural network training
        run_training_pipeline()
    except Exception as e:
        logger.error("FATAL: Pipeline execution failed.")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    # Step 1: Download
    raw_csv = download_kaggle_data()
    
    # Step 2: Extract exactly 7,500 culturally balanced movies
    process_raw_dataset(raw_csv, OUTPUT_CSV_PATH)
    
    # Step 3: Train the math engine
    run_training()
