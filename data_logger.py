import os
from supabase import create_client, Client

URL = os.environ.get("SUPABASE_URL")
KEY = os.environ.get("SUPABASE_KEY")

# Add movie_title as a required parameter (type: str)
def log_user_interaction(username: str, movie_id: int, movie_title: str, rating: float) -> None:
    if not URL or not KEY:
        print("Supabase credentials missing. Logging aborted.")
        return
        
    supabase: Client = create_client(URL, KEY)
    try:
        # Include the new column in your payload
        data = {
            "username": username,
            "movie_id": movie_id,
            "movie_title": movie_title, 
            "rating": rating
        }
        supabase.table("user_interactions").insert(data).execute()
    except Exception as e:
        print(f"Cloud logging failed: {e}")
