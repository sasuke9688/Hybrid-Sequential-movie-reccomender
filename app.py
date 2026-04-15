import os
import traceback
import logging
from functools import wraps

import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template, session
from supabase import create_client, Client

from config import (
    FLASK_HOST, FLASK_PORT, FLASK_SECRET_KEY, TOP_K,
    RATING_SCALE_MAX, MIN_LANGUAGE_COUNT
)

# Import ONLY the lightweight RecommendationEngine
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

# 2. Instantiate Flask App
app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = FLASK_SECRET_KEY

# 3. Initialize Global Supabase Client
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase_client = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)

# 4. Global Engine Initialization (Lightweight Scikit-Learn)
engine = None
engine_error = "No error recorded."

try:
    logger.info("Loading lightweight ML artifacts...")
    tmdb_df = joblib.load("models/tmdb_dataset.pkl")
    mlb = joblib.load("models/mlb.pkl")
    tmdb_latent = joblib.load("models/tmdb_latent.pkl")

    logger.info("Initiating Fast Recommendation Engine...")
    engine = RecommendationEngine(tmdb_latent, tmdb_df, mlb)
    logger.info("Engine instantiated successfully.")
    
except Exception as e:
    engine_error = traceback.format_exc()
    logger.error(f"FATAL: Engine initialization failed.\n{engine_error}")


# 5. Helper Functions
def _parse_genre_filters(raw_value):
    if not raw_value:
        return []
    if isinstance(raw_value, list):
        values = raw_value
    else:
        values = str(raw_value).split(",")
    return [str(value).strip() for value in values if str(value).strip()]

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "username" not in session:
            return jsonify({"error": "Login required"}), 401
        return f(*args, **kwargs)
    return decorated


# 6. Routing Definitions
@app.route('/debug')
def debug_boot():
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
    if not data: return jsonify({"error": "No data provided"}), 400
    username = data.get("username", "").strip()
    password = data.get("password", "")
    ok, msg = register_user(username, password)
    if not ok: return jsonify({"error": msg}), 400
    session["username"] = username.lower()
    return jsonify({"message": msg, "username": username.lower()})

@app.route("/api/login", methods=["POST"])
def api_login():
    data = request.get_json()
    if not data: return jsonify({"error": "No data provided"}), 400
    username = data.get("username", "").strip()
    password = data.get("password", "")
    ok, msg = authenticate_user(username, password)
    if not ok: return jsonify({"error": msg}), 401
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
    ok, msg = add_to_watch_history(session["username"], data["index"], data["title"], data.get("release_year", 0), data.get("rating"))
    if not ok: return jsonify({"error": msg}), 400
    return jsonify({"message": msg})

@app.route("/api/history/<int:movie_index>/rating", methods=["PUT"])
@login_required
def api_update_rating(movie_index):
    data = request.get_json()
    if not data or "rating" not in data:
        return jsonify({"error": "rating required"}), 400
    ok, msg = update_rating(session["username"], movie_index, data["rating"])
    if not ok: return jsonify({"error": msg}), 400
    return jsonify({"message": msg})

@app.route("/api/history/<int:movie_index>", methods=["DELETE"])
@login_required
def api_remove_history(movie_index):
    ok, msg = remove_from_watch_history(session["username"], movie_index)
    if not ok: return jsonify({"error": msg}), 400
    return jsonify({"message": msg})

# --- Catalog API ---
@app.route("/api/languages", methods=["GET"])
def api_languages():
    if engine is None: return jsonify({"error": "Engine unavailable"}), 500
    return jsonify({"languages": engine.get_available_languages(min_count=MIN_LANGUAGE_COUNT)})

@app.route("/api/genres", methods=["GET"])
def api_genres():
    if engine is None: return jsonify({"error": "Engine unavailable"}), 500
    return jsonify({"genres": engine.get_available_genres()})

# --- Recommendation API ---
@app.route("/api/search", methods=["GET"])
def search_movies():
    if not supabase_client: return jsonify({"error": "Database connection missing"}), 500
    query = request.args.get("q", "").strip()
    language = request.args.get("language", "").strip()

    if len(query) < 2 and not language:
        return jsonify({"results": []})

    try:
        db_query = supabase_client.table("tmdb_movies").select("*")
        if query: db_query = db_query.ilike("title", f"%{query}%")
        if language: db_query = db_query.eq("original_language", language)
        response = db_query.limit(50).execute()

        results = []
        for row in response.data:
            # Safely extract the year from the YYYY-MM-DD string
            release_date = str(row.get("release_date", ""))
            release_year = release_date.split("-")[0] if release_date and release_date != "nan" else "N/A"

            results.append({
                "index": int(row.get("pandas_index", 0)), 
                "title": row.get("title", "Unknown Title"),
                "release_year": release_year,
                "genres": row.get("genres", ""),
                "vote_average": row.get("vote_average", "N/A"),
                "popularity": row.get("popularity", "N/A"),
                "original_language": row.get("original_language", "N/A")
            })
        return jsonify({"results": results})
    except Exception as e:
        logger.error(f"Supabase search failed: {e}")
        return jsonify({"error": "Database search failed"}), 500

@app.route("/api/recommend", methods=["POST"])
def recommend():
    if engine is None: return jsonify({"error": "Engine unavailable"}), 500
    data = request.get_json()
    if not data or "movies" not in data: return jsonify({"error": "No movies provided"}), 400

    selected_movies = data["movies"]
    if not selected_movies: return jsonify({"error": "Please select at least one movie"}), 400

    watch_history = get_watch_history(session["username"]) if "username" in session else None

    try:
        recs_df, weight_info = engine.recommend(
            selected_movies,
            top_k=data.get("top_k", TOP_K),
            watch_history=watch_history,
            language_filter=data.get("language", "").strip() or None,
            genre_filters=_parse_genre_filters(data.get("genres", [])),
        )

        # Scrub all 'NaN' values from the Pandas dataframe before JSON conversion
        recs_df = recs_df.fillna("N/A")

        recommendations = recs_df.to_dict(orient="records")
        for rec in recommendations:
            if isinstance(rec["genres"], list):
                rec["genres"] = ", ".join(rec["genres"])

        return jsonify({
            "recommendations": recommendations,
            "weight_info":     weight_info,
            "auto_watched":    False, 
            "rating_prompts":  []
        })

    except Exception as e:
        logger.error(f"Recommendation calculation failed: {e}")
        return jsonify({"error": "Failed to generate recommendations. Please try again."}), 500

@app.route("/api/stats", methods=["GET"])
def stats():
    if engine is None: return jsonify({"error": "Engine unavailable"}), 500
    lang_count = engine.tmdb_df["original_language"].nunique() if "original_language" in engine.tmdb_df.columns else 0
    return jsonify({
        "total_movies":    len(engine.tmdb_df),
        "architecture":    "Fast Cosine Similarity (Scikit-Learn)",
        "language_count":  lang_count,
    })

if __name__ == "__main__":
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=True)
