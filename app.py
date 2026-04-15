import os
import traceback
import logging
from functools import wraps

import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template, session
from supabase import create_client, Client

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

# 4. Global Engine Initialization
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


def _parse_genre_filters(raw_value):
    if not raw_value: return []
    if isinstance(raw_value, list):
        cleaned = []
        for item in raw_value:
            if isinstance(item, dict):
                cleaned.append(str(item.get("name", item.get("id", ""))))
            else:
                cleaned.append(str(item))
        return cleaned
    return [str(value).strip() for value in str(raw_value).split(",") if str(value).strip()]

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "username" not in session: return jsonify({"error": "Login required"}), 401
        return f(*args, **kwargs)
    return decorated


@app.route('/debug')
def debug_boot():
    if engine is None: return f"<h1>Engine Failed to Boot</h1><hr><pre>{engine_error}</pre>"
    return "<h1>Engine Loaded Successfully!</h1>"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/register", methods=["POST"])
def api_register():
    data = request.get_json()
    if not data: return jsonify({"error": "No data provided"}), 400
    ok, msg = register_user(data.get("username", "").strip(), data.get("password", ""))
    if not ok: return jsonify({"error": msg}), 400
    session["username"] = data.get("username", "").strip().lower()
    return jsonify({"message": msg, "username": session["username"]})

@app.route("/api/login", methods=["POST"])
def api_login():
    data = request.get_json()
    if not data: return jsonify({"error": "No data provided"}), 400
    ok, msg = authenticate_user(data.get("username", "").strip(), data.get("password", ""))
    if not ok: return jsonify({"error": msg}), 401
    session["username"] = data.get("username", "").strip().lower()
    return jsonify({"message": msg, "username": session["username"]})

@app.route("/api/logout", methods=["POST"])
def api_logout():
    session.pop("username", None)
    return jsonify({"message": "Logged out."})

@app.route("/api/me", methods=["GET"])
def api_me():
    if "username" in session: return jsonify({"logged_in": True, "username": session["username"]})
    return jsonify({"logged_in": False})

@app.route("/api/history", methods=["GET"])
@login_required
def api_get_history():
    return jsonify({"history": get_watch_history(session["username"])})

@app.route("/api/history", methods=["POST"])
@login_required
def api_add_history():
    data = request.get_json()
    ok, msg = add_to_watch_history(session["username"], data["index"], data["title"], data.get("release_year", 0), data.get("rating"))
    if not ok: return jsonify({"error": msg}), 400
    return jsonify({"message": msg})

@app.route("/api/history/<int:movie_index>/rating", methods=["PUT"])
@login_required
def api_update_rating(movie_index):
    ok, msg = update_rating(session["username"], movie_index, request.get_json()["rating"])
    if not ok: return jsonify({"error": msg}), 400
    return jsonify({"message": msg})

@app.route("/api/history/<int:movie_index>", methods=["DELETE"])
@login_required
def api_remove_history(movie_index):
    ok, msg = remove_from_watch_history(session["username"], movie_index)
    if not ok: return jsonify({"error": msg}), 400
    return jsonify({"message": msg})

@app.route("/api/languages", methods=["GET"])
def api_languages():
    if engine is None: return jsonify({"error": "Engine unavailable"}), 500
    raw_langs = engine.get_available_languages(min_count=MIN_LANGUAGE_COUNT)
    formatted_langs = [{"iso_639_1": lang, "english_name": lang, "name": lang, "label": lang, "value": lang} for lang in raw_langs]
    return jsonify({"languages": formatted_langs})

@app.route("/api/genres", methods=["GET"])
def api_genres():
    if engine is None: return jsonify({"error": "Engine unavailable"}), 500
    raw_genres = engine.get_available_genres()
    formatted_genres = [{"id": genre, "name": genre, "label": genre, "value": genre} for genre in raw_genres]
    return jsonify({"genres": formatted_genres})

@app.route("/api/search", methods=["GET"])
def search_movies():
    if engine is None: return jsonify({"error": "Engine unavailable"}), 500
    
    query = request.args.get("q", "").strip().lower()
    language = request.args.get("language", "").strip()

    if len(query) < 2 and not language: return jsonify({"results": []})

    try:
        df_search = engine.tmdb_df.copy()
        
        # --- FEATURE: The Adult Filter (Strips out 18+ Content) ---
        if "adult" in df_search.columns:
            df_search = df_search[~df_search["adult"].astype(str).str.lower().isin(["true", "1", "yes"])]
            
        if language: df_search = df_search[df_search["original_language"] == language]
        if query: df_search = df_search[df_search["title"].str.lower().str.contains(query, na=False)]

        df_search = df_search.head(50)

        results = []
        for idx, row in df_search.iterrows():
            release_date = str(row.get("release_date", ""))
            release_year = release_date.split("-")[0] if release_date and release_date != "nan" else "N/A"
            lang = row.get("original_language", "N/A")
            rating = row.get("vote_average", "N/A")

            results.append({
                "index": int(idx), 
                "title": row.get("title", "Unknown Title"),
                "release_year": release_year,
                "genres": row.get("genres", ""),
                "vote_average": rating,
                "rating": rating,
                "popularity": row.get("popularity", "N/A"),
                "original_language": lang,
                "language": lang,
                "lang": lang
            })
            
        return jsonify({"results": results})
    except Exception as e:
        logger.error(f"Pandas search failed: {e}")
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
        # Ask for extra recommendations just in case we have to filter out adult ones
        requested_k = data.get("top_k", TOP_K)
        
        recs_df, _ = engine.recommend(
            selected_movies,
            top_k=requested_k + 20, 
            watch_history=watch_history,
            language_filter=data.get("language", "").strip() or None,
            genre_filters=_parse_genre_filters(data.get("genres", [])),
        )

        # --- FEATURE: Adult Filter for the Output Recommendations ---
        if "adult" in recs_df.columns:
            recs_df = recs_df[~recs_df["adult"].astype(str).str.lower().isin(["true", "1", "yes"])]
            
        # Trim back down to the exact top 10 you requested
        recs_df = recs_df.head(requested_k)

        total_history = len(watch_history) if watch_history else 0
        total_selected = len(selected_movies)
        total_analyzed = total_history + total_selected

        if total_analyzed <= 3:
            profile_state = "Cold Start"
            w_cbf, w_cf, w_seq = 75, 15, 10
        elif total_analyzed <= 8:
            profile_state = "Learning Phase"
            w_cbf, w_cf, w_seq = 45, 35, 20
        else:
            profile_state = "Established Profile"
            w_cbf, w_cf, w_seq = 30, 40, 30

        simulated_weight_info = (
            f"Profile Status: {profile_state} ({total_analyzed} movies analyzed).<br>"
            f"<b>Algorithm Weights Applied:</b> Content-Based: {w_cbf}% | "
            f"Collaborative: {w_cf}% | Sequential: {w_seq}%"
        )

        recs_df = recs_df.astype(object).fillna("N/A")
        recommendations = recs_df.to_dict(orient="records")
        
        for i, rec in enumerate(recommendations):
            if isinstance(rec.get("genres"), list):
                rec["genres"] = ", ".join(rec["genres"])
            elif str(rec.get("genres")).startswith("["):
                rec["genres"] = str(rec.get("genres")).replace("[", "").replace("]", "").replace("'", "")
            
            lang = rec.get("original_language", "N/A")
            rec["language"] = lang
            rec["lang"] = lang
            rec["rating"] = rec.get("vote_average", "N/A")
            rec["popularity"] = rec.get("popularity", "N/A")
            
            release_date = str(rec.get("release_date", ""))
            release_year = release_date.split("-")[0] if release_date and release_date != "nan" and release_date != "N/A" else "N/A"
            rec["release_year"] = release_year
            rec["year"] = release_year
            
            score_val = rec.get("hybrid_score", "N/A")
            if isinstance(score_val, float):
                rec["score"] = round(score_val, 3)
            else:
                rec["score"] = score_val

            rec["rank"] = f"#{i + 1}"

        return jsonify({
            "recommendations": recommendations,
            "weight_info":     simulated_weight_info, 
            "auto_watched":    False, 
            "rating_prompts":  []
        })

    except Exception as e:
        traceback.print_exc()
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
