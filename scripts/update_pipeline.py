import os
import sys
import logging
import pandas as pd
import zipfile

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
KAGGLE_DATASET = "asaniczka/tmdb-movies-dataset-2023-930k-movies"
# --- THE FIX: Output exactly to the filename your training script expects ---
OUTPUT_CSV_PATH = "golden_tmdb_11k.csv" 

def download_kaggle_data():
    logger.info("Authenticating with Kaggle API...")
    try:
        import kaggle
        logger.info("Downloading massive raw TMDB dataset...")
        kaggle.api.dataset_download_files(KAGGLE_DATASET, path='.', unzip=True)
        
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
    logger.info("Loading raw dataset to extract the 11.5k Golden TMDB distribution...")
    
    df_raw = pd.read_csv(raw_csv_path, low_memory=False)
    
    # 1. Quality threshold
    if 'vote_count' in df_raw.columns:
        df_filtered = df_raw[df_raw['vote_count'] >= 50].copy()
    else:
        df_filtered = df_raw.copy()
        
    # 2. Sort by highest rated/most popular
    if 'popularity' in df_filtered.columns:
        df_filtered = df_filtered.sort_values(by='popularity', ascending=False)
        
    # 3. Stratified Extraction: 11,500 Total Movies
    df_en = df_filtered[df_filtered['original_language'] == 'en'].head(5000)
    df_ja = df_filtered[df_filtered['original_language'] == 'ja'].head(1500)
    
    indian_langs = ['hi', 'te', 'ta', 'ml']
    df_in = df_filtered[df_filtered['original_language'].isin(indian_langs)].head(5000)
    
    # 4. Concatenate and Shuffle
    df_final = pd.concat([df_en, df_ja, df_in], ignore_index=True)
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 5. Save as golden_tmdb_11k.csv
    logger.info(f"Writing Golden dataset ({len(df_final)} rows) to {output_csv_path}...")
    df_final.to_csv(output_csv_path, index=False)
    logger.info("Golden Dataset Extraction Complete.")

def run_training():
    logger.info("Initiating core hybrid training sequence...")
    try:
        # Connect to your model_training.py script
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from model_training import run_training_pipeline
        
        run_training_pipeline()
    except Exception as e:
        logger.error("FATAL: Pipeline execution failed.")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    # Step 1: Download latest Kaggle data
    raw_csv = download_kaggle_data()
    
    # Step 2: Build the Golden 11.5k Dataset
    process_raw_dataset(raw_csv, OUTPUT_CSV_PATH)
    
    # Step 3: Train the PyTorch engine
    run_training()
