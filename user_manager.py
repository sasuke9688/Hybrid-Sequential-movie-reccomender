import os
import logging
from werkzeug.security import generate_password_hash, check_password_hash
from supabase import create_client, Client

logger = logging.getLogger(__name__)

# Initialize Supabase Client
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

supabase = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
else:
    logger.error("FATAL: Supabase credentials missing in user_manager.py")

def register_user(username, password):
    if not supabase: return False, "Database connection error"
    username = username.strip().lower()
    
    try:
        # Check if user already exists
        res = supabase.table("app_users").select("*").eq("username", username).execute()
        if len(res.data) > 0:
            return False, "Username already exists"
            
        # Hash password and save to Supabase
        hashed_pw = generate_password_hash(password)
        supabase.table("app_users").insert({"username": username, "password_hash": hashed_pw}).execute()
        return True, "Registration successful"
    except Exception as e:
        logger.error(f"DB Error during registration: {e}")
        return False, "Failed to register account"

def authenticate_user(username, password):
    if not supabase: return False, "Database connection error"
    username = username.strip().lower()
    
    try:
        # Fetch user from Supabase
        res = supabase.table("app_users").select("*").eq("username", username).execute()
        if len(res.data) == 0:
            return False, "Invalid username or password"
            
        user = res.data[0]
        # Verify Password
        if check_password_hash(user["password_hash"], password):
            return True, "Login successful"
        return False, "Invalid username or password"
    except Exception as e:
        logger.error(f"DB Error during login: {e}")
        return False, "Login validation failed"

def get_watch_history(username):
    if not supabase: return []
    try:
        # Fetch all saved movies for this user
        res = supabase.table("watch_history").select("*").eq("username", username).order("added_at", desc=True).execute()
        
        # Format the data exactly how your frontend app.js expects it
        history = []
        for row in res.data:
            history.append({
                "index": row["movie_index"],
                "title": row["title"],
                "release_year": row.get("release_year", "N/A"),
                "rating": row.get("rating")
            })
        return history
    except Exception as e:
        logger.error(f"DB Error fetching history: {e}")
        return []

def add_to_watch_history(username, movie_index, title, release_year, rating=None):
    if not supabase: return False, "Database connection error"
    
    try:
        # Prevent duplicate entries in the user's history
        res = supabase.table("watch_history").select("*").eq("username", username).eq("movie_index", movie_index).execute()
        if len(res.data) > 0:
            return False, "Movie is already in your Watch History!"
            
        data = {
            "username": username,
            "movie_index": int(movie_index),
            "title": str(title),
            "release_year": str(release_year),
            "rating": float(rating) if rating else None
        }
        
        # Save movie directly into the Supabase database
        supabase.table("watch_history").insert(data).execute()
        return True, "Movie added to Watch History"
    except Exception as e:
        logger.error(f"DB Error adding to history: {e}")
        return False, "Failed to save movie"

def update_rating(username, movie_index, rating):
    if not supabase: return False, "Database connection error"
    try:
        supabase.table("watch_history").update({"rating": float(rating)}).eq("username", username).eq("movie_index", int(movie_index)).execute()
        return True, "Rating successfully updated"
    except Exception as e:
        logger.error(f"DB Error updating rating: {e}")
        return False, "Failed to update rating"

def remove_from_watch_history(username, movie_index):
    if not supabase: return False, "Database connection error"
    try:
        supabase.table("watch_history").delete().eq("username", username).eq("movie_index", int(movie_index)).execute()
        return True, "Movie removed from history"
    except Exception as e:
        logger.error(f"DB Error removing history: {e}")
        return False, "Failed to remove movie"
