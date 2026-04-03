# data_logger.py
import os
from supabase import create_client, Client

# Use environment variables so you don't hardcode secrets in GitHub
URL = os.environ.get("https://supabase.com/dashboard/project/niqemviobrituppewxco/editor/17539?schema=public")
KEY = os.environ.get("sbp_a09f1c78d3481ad673bdc42af01970ca7c4e39d9")

def log_user_interaction(username: str, movie_id: int, rating: float) -> None:
    if not URL or not KEY:
        print("Supabase credentials missing. Logging aborted.")
        return
        
    supabase: Client = create_client(URL, KEY)
    try:
        data = {
            "username": username,
            "movie_id": movie_id,
            "rating": rating
        }
        # Insert the row into the cloud database
        supabase.table("user_interactions").insert(data).execute()
    except Exception as e:
        print(f"Cloud logging failed: {e}")