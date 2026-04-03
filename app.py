import pandas as pd
import joblib 
import logging
import traceback
from recommendation_engine import RecommendationEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

engine = None

try:
    logger.info("Loading ML artifacts and dataset...")
    
    # 1. Load the pre-processed dataframe
    tmdb_df = pd.read_csv("data/tmdb_movies.csv")
    
    # 2. Load the serialized machine learning models
    # Ensure these paths point to where your train.py saved them
    tmdb_latent = joblib.load("models/tmdb_latent.pkl")
    mlb = joblib.load("models/mlb.pkl")
    ridge = joblib.load("models/ridge.pkl")
    
    # Optional: Load collaborative filtering factors if utilized
    # user_factors = joblib.load("models/user_factors.pkl")
    # movie_factors = joblib.load("models/movie_factors.pkl")

    logger.info("Initiating Hybrid Recommendation Engine...")
    
    # 3. Instantiate the engine with the required objects
    engine = RecommendationEngine(
        tmdb_df=tmdb_df,
        tmdb_latent=tmdb_latent,
        mlb=mlb,
        ridge=ridge
        # user_factors=user_factors,
        # movie_factors=movie_factors
    )
    
    logger.info("Engine instantiated successfully.")
except Exception as e:
    logger.error("FATAL: Engine initialization failed.")
    logger.error(traceback.format_exc())
# ... rest of your Flask app code ...
import os
from functools import wraps
from flask import Flask, request, jsonify, render_template, session
import joblib
# app.py (Top of file)
import os
from functools import wraps
from flask import Flask, request, jsonify, render_template, session
import joblib

# ... existing imports ...
from user_manager import (
    register_user, authenticate_user,
    add_to_watch_history, remove_from_watch_history, get_watch_history,
    update_rating
)
from data_logger import log_user_interaction  # <-- NEW IMPORT
from config import (
    MODEL_DIR, FLASK_HOST, FLASK_PORT, FLASK_SECRET_KEY, TOP_K,
    RATING_SCALE_MAX, MIN_LANGUAGE_COUNT
)
from recommendation_engine import RecommendationEngine
from user_manager import (
    register_user, authenticate_user,
    add_to_watch_history, remove_from_watch_history, get_watch_history,
    update_rating
)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = FLASK_SECRET_KEY

# Global engine instance
engine = None


def _parse_genre_filters(raw_value):
    """Parse a genre filter from query/body data into a clean list."""
    if not raw_value:
        return []

    if isinstance(raw_value, list):
        values = raw_value
    else:
        values = str(raw_value).split(",")

    return [str(value).strip() for value in values if str(value).strip()]


def load_engine():
    """Load model artifacts and initialize the recommendation engine."""
    global engine

    print("Loading model artifacts...")
    tmdb_df     = joblib.load(os.path.join(MODEL_DIR, "tmdb_dataset.pkl"))
    tmdb_latent = joblib.load(os.path.join(MODEL_DIR, "tmdb_latent.pkl"))
    mlb         = joblib.load(os.path.join(MODEL_DIR, "mlb.pkl"))
    ridge       = joblib.load(os.path.join(MODEL_DIR, "ridge_model.pkl"))
    user_factors = joblib.load(os.path.join(MODEL_DIR, "user_factors.pkl"))
    movie_factors = joblib.load(os.path.join(MODEL_DIR, "movie_factors.pkl"))

    engine = RecommendationEngine(
        tmdb_df=tmdb_df,
        tmdb_latent=tmdb_latent,
        mlb=mlb,
        ridge=ridge,
        user_factors=user_factors,
        movie_factors=movie_factors,
    )
    print(f"Engine loaded. TMDB catalog: {len(tmdb_df)} movies")


def login_required(f):
    """Decorator to require login for API endpoints."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if "username" not in session:
            return jsonify({"error": "Login required"}), 401
        return f(*args, **kwargs)
    return decorated


# ──────────────────────────── Pages ────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


# ──────────────────────────── Auth API ────────────────────────────

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


# ──────────────────────────── Watch History API ────────────────────────────

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

    ok, msg = add_to_watch_history(
        session["username"], movie_index, movie_title, release_year, rating
    )
    if not ok:
        return jsonify({"error": msg}), 400

    # <-- ADD THE FOLLOWING BLOCK -->
    if rating is not None:
        try:
            log_user_interaction(session["username"], movie_index, rating)
        except Exception as e:
            print(f"Data logging failed for user {session['username']}: {e}")
    # <----------------------------->

    
    return jsonify({"message": msg})


@app.route("/api/history/<int:movie_index>/rating", methods=["PUT"])
@login_required
def api_update_rating(movie_index):
    data = request.get_json()
    if not data or "rating" not in data:
        return jsonify({"error": "rating required"}), 400

    # Ensure the variable is explicitly defined from the payload
    rating = data["rating"]
    
    if not isinstance(rating, (int, float)) or rating < 1 or rating > RATING_SCALE_MAX:
        return jsonify({"error": f"Rating must be between 1 and {RATING_SCALE_MAX}"}), 400

    ok, msg = update_rating(session["username"], movie_index, rating)
    if not ok:
        return jsonify({"error": msg}), 400

    # Execute Supabase logging
    try:
        log_user_interaction(session["username"], movie_index, rating)
    except Exception as e:
        print(f"Data logging failed for user {session['username']}: {e}")

    return jsonify({"message": msg})
@app.route("/api/history/<int:movie_index>", methods=["DELETE"])
@login_required
def api_remove_history(movie_index):
    ok, msg = remove_from_watch_history(session["username"], movie_index)
    if not ok:
        return jsonify({"error": msg}), 400
    return jsonify({"message": msg})


# ──────────────────────────── Language API (NEW) ────────────────────────────

@app.route("/api/languages", methods=["GET"])
def api_languages():
    """
    Return the list of languages present in the TMDB catalog.
    Each entry: { code, label, count }
    Sorted by movie count descending.
    """
    languages = engine.get_available_languages(min_count=MIN_LANGUAGE_COUNT)
    return jsonify({"languages": languages})


@app.route("/api/genres", methods=["GET"])
def api_genres():
    """Return the list of genres present in the TMDB catalog."""
    genres = engine.get_available_genres()
    return jsonify({"genres": genres})


# ──────────────────────────── Movie Search & Recommend ────────────────────────────

@app.route("/api/search", methods=["GET"])
def search_movies():
    """
    Search for movies by title.
    Optional query param: ?language=en  to restrict results to one language.
    """
    query    = request.args.get("q", "").strip()
    language = request.args.get("language", "").strip()  # NEW
    genres   = _parse_genre_filters(request.args.get("genres", ""))

    if not query or len(query) < 2:
        return jsonify({"results": []})

    results = engine.search_movies(
        query,
        limit=20,
        language_filter=language if language else None,   # NEW
        genre_filters=genres,
    )
    return jsonify({"results": results})


@app.route("/api/recommend", methods=["POST"])
def recommend():
    """
    Generate recommendations based on selected movies + user history.
    """
    data = request.get_json()
    if not data or "movies" not in data:
        return jsonify({"error": "No movies provided"}), 400

    selected_movies = data["movies"]
    top_k           = data.get("top_k", TOP_K)
    language        = data.get("language", "").strip()
    genres          = _parse_genre_filters(data.get("genres", []))

    if not selected_movies:
        return jsonify({"error": "Please select at least one movie"}), 400

    # Validate movie indices
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

    # ── Auto-mark selected movies as watched and log interactions ────────
    # (Removed the rating_prompts list to prevent the double-asking UI bug)
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

            # --- DATA LOGGING INTEGRATION ---
            if ok and rating is not None:
                try:
                    log_user_interaction(session["username"], idx, rating)
                except Exception as e:
                    print(f"Data logging failed for user {session['username']}: {e}")
            # --------------------------------
    # ──────────────────────────────────────────────────────────────────────

    # Load user watch history for the engine
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

        # Convert genre lists to strings for display
        for rec in recommendations:
            if isinstance(rec["genres"], list):
                rec["genres"] = ", ".join(rec["genres"])

        return jsonify({
            "recommendations": recommendations,
            "weight_info":     weight_info,
            "rating_prompts":  [],      # Hardcoded to empty so frontend doesn't prompt again
            "auto_watched":    False,   # Disabled to stop the second popup
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    """
    Generate recommendations based on selected movies + user history.

    Request body:
      {
        "movies": [{"index": 0, "title": "...", "release_year": 2020, "rating": 4}],
        "top_k": 10,
        "language": "en",         # optional language filter
        "genres": ["Action"]      # optional genre filters
      }
    """
    data = request.get_json()
    if not data or "movies" not in data:
        return jsonify({"error": "No movies provided"}), 400

    selected_movies = data["movies"]
    top_k           = data.get("top_k", TOP_K)
    language        = data.get("language", "").strip()
    genres          = _parse_genre_filters(data.get("genres", []))

    if not selected_movies:
        return jsonify({"error": "Please select at least one movie"}), 400

    # Validate movie indices
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

    # ── NEW: Auto-mark selected movies as watched and log interactions ────────
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

            # --- DATA LOGGING INTEGRATION ---
            if ok and rating is not None:
                try:
                    log_user_interaction(session["username"], idx, rating)
                except Exception as e:
                    # Non-blocking error handling to ensure API continuity
                    print(f"Data logging failed for user {session['username']}: {e}")
            # --------------------------------

            # Collect movies that have no rating yet so the frontend can prompt
            if ok and (rating is None):
                rating_prompts.append({
                    "index":        idx,
                    "title":        movie_title,
                    "release_year": release_year,
                })
    # ──────────────────────────────────────────────────────────────────────

    # Load user watch history for the engine
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

        # Convert genre lists to strings for display
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
# ──────────────────────────── Stats ────────────────────────────

@app.route("/api/stats", methods=["GET"])
def stats():
    lang_count = 0
    if "original_language" in engine.tmdb_df.columns:
        lang_count = engine.tmdb_df["original_language"].nunique()

    return jsonify({
        "total_movies":    len(engine.tmdb_df),
        "latent_dim":      engine.tmdb_latent.shape[1],
        "language_count":  lang_count,          # NEW
    })


if __name__ == "__main__":
    load_engine()
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=True)
