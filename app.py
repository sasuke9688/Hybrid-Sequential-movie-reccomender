import os
import traceback
import logging
from functools import wraps

import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template, session

from config import (
    FLASK_HOST, FLASK_PORT, FLASK_SECRET_KEY, TOP_K,
    RATING_SCALE_MAX, MIN_LANGUAGE_COUNT
)
from recommendation_engine import RecommendationEngine
from user_manager import (
    register_user, authenticate_user,
    add_to_watch_history, remove_from_watch_history, get_watch_history,
    update_rating
)
from data_logger import log_user_interaction

# 1. Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 2. Instantiate Flask App (Must occur before ANY @app.route decorators)
app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = FLASK_SECRET_KEY

# 3. Global Engine Initialization
engine = None
engine_error = "No error recorded."

try:
    logger.info("Loading ML artifacts and dataset...")
    # Load the lightweight pre-processed dataframe
    # Load the fully processed dataframe artifact
    tmdb_df = joblib.load("models/tmdb_dataset.pkl")
    
    # Load the serialized machine learning matrices
    tmdb_latent = joblib.load("models/tmdb_latent.pkl")
    mlb = joblib.load("models/mlb.pkl")
    ridge = joblib.load("models/ridge_model.pkl")

    logger.info("Initiating Hybrid Recommendation Engine...")
    engine = RecommendationEngine(
        tmdb_df=tmdb_df,
        tmdb_latent=tmdb_latent,
        mlb=mlb,
        ridge=ridge
    )
    logger.info("Engine instantiated successfully.")
except Exception as e:
    engine_error = traceback.format_exc()
    logger.error(f"FATAL: Engine initialization failed.\n{engine_error}")


# 4. Helper Functions
def _parse_genre_filters(raw_value):
    """Parse a genre filter from query/body data into a clean list."""
    if not raw_value:
        return []
    if isinstance(raw_value, list):
        values = raw_value
    else:
        values = str(raw_value).split(",")
    return [str(value).strip() for value in values if str(value).strip()]

def login_required(f):
    """Decorator to require login for API endpoints."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if "username" not in session:
            return jsonify({"error": "Login required"}), 401
        return f(*args, **kwargs)
    return decorated


# 5. Routing Definitions

@app.route('/debug')
def debug_boot():
    """Diagnostic route to display startup tracebacks."""
    if engine is None:
        return f"<h1>Engine Failed to Boot</h1><hr><pre>{engine_error}</pre>"
    return "<h1>Engine Loaded Successfully!</h1>"

@app.route("/")
def index():
    return render_template("index.html")

# --- Auth API ---

@app.route("/api/register", methods=["POST"])
def api_register():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    username = data.get("username", "").strip()
    password = data.get("password", "")

    ok, msg = register_user(username, password)
    if not ok:
        return jsonify({"error": msg}), 400

    session["username"] = username.lower()
    return jsonify({"message": msg, "username": username.lower()})

@app.route("/api/login", methods=["POST"])
def api_login():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    username = data.get("username", "").strip()
    password = data.get("password", "")

    ok, msg = authenticate_user(username, password)
    if not ok:
        return jsonify({"error": msg}), 401

    session["username"] = username.lower()
    return jsonify({"message": msg, "username": username.lower()})

@app.route("/api/logout", methods=["POST"])
def api_logout():
    session.pop("username", None)
    return jsonify({"message": "Logged out."})

@app.route("/api/me", methods=["GET"])
def api_me():
    if "username" in session:
        return jsonify({"logged_in": True, "username": session["username"]})
    return jsonify({"logged_in": False})

# --- Watch History API ---

@app.route("/api/history", methods=["GET"])
@login_required
def api_get_history():
    history = get_watch_history(session["username"])
    return jsonify({"history": history})

@app.route("/api/history", methods=["POST"])
@login_required
def api_add_history():
    data = request.get_json()
    if not data or "index" not in data or "title" not in data:
        return jsonify({"error": "index and title required"}), 400

    movie_index  = data["index"]
    movie_title  = data["title"]
    release_year = data.get("release_year", 0)
    rating       = data.get("rating")

    if not isinstance(movie_index, int) or movie_index < 0 or movie_index >= len(engine.tmdb_df):
        return jsonify({"error": f"Invalid movie index: {movie_index}"}), 400

    if rating is not None:
        if not isinstance(rating, (int, float)) or rating < 1 or rating > RATING_SCALE_MAX:
            return jsonify({"error": f"Rating must be between 1 and {RATING_SCALE_MAX}"}), 400

    ok, msg = add_to_watch_history(
        session["username"], movie_index, movie_title, release_year, rating
    )
    if not ok:
        return jsonify({"error": msg}), 400

    if rating is not None:
        try:
            # UPDATE THIS LINE: Add movie_title
            log_user_interaction(session["username"], movie_index, movie_title, rating)
        except Exception as e:
            logger.error(f"Data logging failed for user {session['username']}: {e}")

    return jsonify({"message": msg})

@app.route("/api/history/<int:movie_index>/rating", methods=["PUT"])
@login_required
def api_update_rating(movie_index):
    data = request.get_json()
    if not data or "rating" not in data:
        return jsonify({"error": "rating required"}), 400

    rating = data["rating"]
    if not isinstance(rating, (int, float)) or rating < 1 or rating > RATING_SCALE_MAX:
        return jsonify({"error": f"Rating must be between 1 and {RATING_SCALE_MAX}"}), 400

    ok, msg = update_rating(session["username"], movie_index, rating)
    if not ok:
        return jsonify({"error": msg}), 400

   try:
        # UPDATE THIS BLOCK: Fetch the title from the engine, then log it
        movie_title = engine.tmdb_df.iloc[movie_index]["title"]
        log_user_interaction(session["username"], movie_index, movie_title, rating)
    except Exception as e:
        logger.error(f"Data logging failed for user {session['username']}: {e}")

    return jsonify({"message": msg})

@app.route("/api/history/<int:movie_index>", methods=["DELETE"])
@login_required
def api_remove_history(movie_index):
    ok, msg = remove_from_watch_history(session["username"], movie_index)
    if not ok:
        return jsonify({"error": msg}), 400
    return jsonify({"message": msg})

# --- Catalog API ---

@app.route("/api/languages", methods=["GET"])
def api_languages():
    if engine is None:
         return jsonify({"error": "Engine unavailable"}), 500
    languages = engine.get_available_languages(min_count=MIN_LANGUAGE_COUNT)
    return jsonify({"languages": languages})

@app.route("/api/genres", methods=["GET"])
def api_genres():
    if engine is None:
         return jsonify({"error": "Engine unavailable"}), 500
    genres = engine.get_available_genres()
    return jsonify({"genres": genres})

# --- Recommendation API ---

@app.route("/api/search", methods=["GET"])
def search_movies():
    if engine is None:
         return jsonify({"error": "Engine unavailable"}), 500
         
    query    = request.args.get("q", "").strip()
    language = request.args.get("language", "").strip()
    genres   = _parse_genre_filters(request.args.get("genres", ""))

    if not query or len(query) < 2:
        return jsonify({"results": []})

    results = engine.search_movies(
        query,
        limit=20,
        language_filter=language if language else None,
        genre_filters=genres,
    )
    return jsonify({"results": results})

@app.route("/api/recommend", methods=["POST"])
def recommend():
    if engine is None:
         return jsonify({"error": "Engine unavailable"}), 500

    data = request.get_json()
    if not data or "movies" not in data:
        return jsonify({"error": "No movies provided"}), 400

    selected_movies = data["movies"]
    top_k           = data.get("top_k", TOP_K)
    language        = data.get("language", "").strip()
    genres          = _parse_genre_filters(data.get("genres", []))

    if not selected_movies:
        return jsonify({"error": "Please select at least one movie"}), 400

    for movie in selected_movies:
        if "index" not in movie:
            return jsonify({"error": "Each movie must have an 'index' field"}), 400
        idx = movie["index"]
        if not isinstance(idx, int) or idx < 0 or idx >= len(engine.tmdb_df):
            return jsonify({"error": f"Invalid movie index: {idx}"}), 400
        rating = movie.get("rating")
        if rating is not None:
            if not isinstance(rating, (int, float)) or rating < 1 or rating > RATING_SCALE_MAX:
                return jsonify({"error": f"Movie ratings must be between 1 and {RATING_SCALE_MAX}"}), 400

    rating_prompts = []  
    if "username" in session:
        for movie in selected_movies:
            idx          = movie["index"]
            row          = engine.tmdb_df.iloc[idx]
            movie_title  = row["title"]
            release_year = int(row["release_year"])
            rating       = movie.get("rating")

            ok, _ = add_to_watch_history(
                session["username"],
                idx,
                movie_title,
                release_year,
                rating,
            )

           if ok and rating is not None:
                try:
                    # UPDATE THIS LINE: Add movie_title
                    log_user_interaction(session["username"], idx, movie_title, rating)
                except Exception as e:
                    logger.error(f"Data logging failed for user {session['username']}: {e}")

            if ok and (rating is None):
                rating_prompts.append({
                    "index":        idx,
                    "title":        movie_title,
                    "release_year": release_year,
                })

    watch_history = None
    if "username" in session:
        watch_history = get_watch_history(session["username"])

    try:
        recs_df, weight_info = engine.recommend(
            selected_movies,
            top_k=top_k,
            watch_history=watch_history,
            language_filter=language if language else None,
            genre_filters=genres,
        )

        recommendations = recs_df.to_dict(orient="records")

        for rec in recommendations:
            if isinstance(rec["genres"], list):
                rec["genres"] = ", ".join(rec["genres"])

        return jsonify({
            "recommendations": recommendations,
            "weight_info":     weight_info,
            "rating_prompts":  rating_prompts, 
            "auto_watched":    len(rating_prompts) > 0,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/stats", methods=["GET"])
def stats():
    if engine is None:
         return jsonify({"error": "Engine unavailable"}), 500
         
    lang_count = 0
    if "original_language" in engine.tmdb_df.columns:
        lang_count = engine.tmdb_df["original_language"].nunique()

    return jsonify({
        "total_movies":    len(engine.tmdb_df),
        "latent_dim":      engine.tmdb_latent.shape[1],
        "language_count":  lang_count,
    })

if __name__ == "__main__":
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=True)
