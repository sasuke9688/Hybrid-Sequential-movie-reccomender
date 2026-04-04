import os
from supabase import create_client, Client

# Correct usage: look up the NAMES of the variables, not the values!
URL = os.environ.get("SUPABASE_URL")
KEY = os.environ.get("SUPABASE_KEY")

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
