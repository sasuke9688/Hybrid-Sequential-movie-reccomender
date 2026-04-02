"""
Main training script.
Downloads datasets if needed, runs the full training pipeline, and evaluates the model.

The TMDB dataset is fetched from Kaggle (updated daily) to ensure the latest movies
are always included in the recommendation catalog.
"""

import os
import sys
import glob
import zipfile
import subprocess
import requests

from config import DATA_DIR, MODEL_DIR, MOVIELENS_RATINGS, TMDB_MOVIES


MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
KAGGLE_TMDB_DATASET = "asaniczka/tmdb-movies-dataset-2023-930k-movies"


def download_movielens(data_dir=DATA_DIR):
    """Download and extract MovieLens 1M dataset if not present."""
    ml_dir = os.path.join(data_dir, "ml-1m")
    if os.path.exists(os.path.join(ml_dir, "ratings.dat")):
        print("MovieLens 1M dataset already present.")
        return

    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "ml-1m.zip")

    print("Downloading MovieLens 1M dataset...")
    response = requests.get(MOVIELENS_URL, stream=True)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    downloaded = 0

    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(f"\r  Downloaded: {pct:.1f}%", end="", flush=True)

    print("\n  Extracting...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(data_dir)

    os.remove(zip_path)
    print("  MovieLens 1M dataset ready.")


def download_tmdb_from_kaggle(data_dir=DATA_DIR):
    """
    Download the latest TMDB dataset from Kaggle using the kaggle CLI.
    This dataset is updated daily, so it always contains the newest movies.

    Requires:
      - pip install kaggle
      - ~/.kaggle/kaggle.json with valid API credentials
        (get from https://www.kaggle.com/settings -> "Create New Token")
    """
    os.makedirs(data_dir, exist_ok=True)

    print(f"  Downloading latest TMDB dataset from Kaggle...")
    print(f"  Dataset: {KAGGLE_TMDB_DATASET}")

    try:
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", KAGGLE_TMDB_DATASET,
             "-p", data_dir, "--unzip", "--force"],
            capture_output=True, text=True, timeout=300
        )

        if result.returncode != 0:
            stderr = result.stderr.strip()
            if "kaggle.json" in stderr or "Could not find" in stderr or "403" in stderr:
                print("\n  ERROR: Kaggle API credentials not configured.")
                print("  To set up Kaggle API credentials:")
                print("    1. Go to https://www.kaggle.com/settings")
                print("    2. Click 'Create New Token' to download kaggle.json")
                print("    3. Place kaggle.json in ~/.kaggle/ (Linux/Mac)")
                print("       or C:\\Users\\<username>\\.kaggle\\ (Windows)")
                return False
            print(f"\n  Kaggle CLI error: {stderr}")
            return False

        print(f"  {result.stdout.strip()}")

        # Find the downloaded CSV file and rename to expected path
        csv_candidates = glob.glob(os.path.join(data_dir, "*.csv"))
        tmdb_candidates = [f for f in csv_candidates
                           if "tmdb" in os.path.basename(f).lower()
                           or "movie" in os.path.basename(f).lower()]

        target_path = TMDB_MOVIES
        if tmdb_candidates and not os.path.exists(target_path):
            source = tmdb_candidates[0]
            os.rename(source, target_path)
            print(f"  Renamed: {os.path.basename(source)} -> {os.path.basename(target_path)}")
        elif os.path.exists(target_path):
            print(f"  TMDB dataset ready at: {target_path}")
        elif csv_candidates:
            # Use the first CSV found
            source = csv_candidates[0]
            os.rename(source, target_path)
            print(f"  Renamed: {os.path.basename(source)} -> {os.path.basename(target_path)}")
        else:
            print("  WARNING: No CSV file found after download.")
            return False

        return True

    except FileNotFoundError:
        print("\n  ERROR: kaggle CLI not found. Install it with:")
        print("    pip install kaggle")
        return False
    except subprocess.TimeoutExpired:
        print("\n  ERROR: Download timed out. Try again or download manually.")
        return False


def check_tmdb_dataset():
    """Check if TMDB dataset is present, attempt Kaggle download if not."""
    if os.path.exists(TMDB_MOVIES):
        return True

    print(f"\n  TMDB dataset not found at: {TMDB_MOVIES}")
    print("  Attempting to download from Kaggle (updated daily)...")

    if download_tmdb_from_kaggle():
        return os.path.exists(TMDB_MOVIES)

    print(f"\n  Could not auto-download TMDB dataset.")
    print("  Please download it manually from Kaggle:")
    print("    https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies")
    print(f"  and place the CSV file at: {TMDB_MOVIES}")
    return False


def main():
    """Main entry point."""
    print("=" * 60)
    print("  HYBRID MOVIE RECOMMENDATION SYSTEM")
    print("  Setup and Training")
    print("=" * 60)

    # Step 1: Download MovieLens
    print("\n[1/4] Checking datasets...")
    download_movielens()

    if not check_tmdb_dataset():
        sys.exit(1)

    # Step 2: Run training pipeline
    print("\n[2/4] Running training pipeline...")
    from model_training import run_training_pipeline
    artifacts = run_training_pipeline()

    # Step 3: Quick evaluation
    print("\n[3/4] Running quick evaluation...")
    run_quick_evaluation(artifacts)

    # Step 4: Summary
    print("\n[4/4] Setup complete!")
    print(f"\nModel files saved to: {MODEL_DIR}/")
    print("To start the web application, run:")
    print("  python app.py")
    print(f"\nThen open http://localhost:5000 in your browser.")


def run_quick_evaluation(artifacts):
    """Run a quick evaluation by simulating user selections."""
    from recommendation_engine import RecommendationEngine
    from evaluation import precision_at_k, recall_at_k, ndcg_at_k

    tmdb_df = artifacts["tmdb_df"]
    tmdb_latent = artifacts["tmdb_latent"]
    mlb = artifacts["mlb"]
    ridge = artifacts["ridge"]

    engine = RecommendationEngine(
        tmdb_df=tmdb_df,
        tmdb_latent=tmdb_latent,
        mlb=mlb,
        ridge=ridge,
    )

    # Simulate: pick some popular Sci-Fi movies and check recommendations
    test_queries = ["interstellar", "inception", "the matrix"]
    selected = []

    for query in test_queries:
        results = engine.search_movies(query, limit=1)
        if results:
            selected.append(results[0])
            print(f"  Selected: {results[0]['title']} ({results[0]['release_year']})")

    if len(selected) >= 2:
        recs, _ = engine.recommend(selected, top_k=10)
        print(f"\n  Top 10 recommendations:")
        for _, row in recs.iterrows():
            genres = row["genres"] if isinstance(row["genres"], str) else ", ".join(row["genres"])
            print(f"    {row['rank']}. {row['title']} ({row['release_year']}) "
                  f"[{genres}] Score: {row['score']}")

        # Basic sanity metrics (using selected movies as pseudo-relevant)
        rec_indices = list(range(len(recs)))
        relevant = set(range(min(3, len(recs))))
        p = precision_at_k(rec_indices, relevant, k=10)
        r = recall_at_k(rec_indices, relevant, k=10)
        n = ndcg_at_k(rec_indices, relevant, k=10)
        print(f"\n  Sanity check metrics (K=10):")
        print(f"    Precision@10: {p:.4f}")
        print(f"    Recall@10:    {r:.4f}")
        print(f"    NDCG@10:      {n:.4f}")
    else:
        print("  Could not find enough test movies for evaluation.")


if __name__ == "__main__":
    main()
