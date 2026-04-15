import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class RecommendationEngine:
    def __init__(self, tmdb_latent, tmdb_df, mlb):
        self.tmdb_latent = tmdb_latent
        self.tmdb_df = tmdb_df
        self.mlb = mlb

    def get_available_languages(self, min_count=5):
        if "original_language" in self.tmdb_df.columns:
            counts = self.tmdb_df["original_language"].value_counts()
            return counts[counts >= min_count].index.tolist()
        return []

    def get_available_genres(self):
        if hasattr(self.mlb, "classes_"):
            return list(self.mlb.classes_)
        return []

    def recommend(self, selected_movies, top_k=10, watch_history=None, language_filter=None, genre_filters=None):
        # 1. Get indices of selected movies
        selected_indices = [m["index"] for m in selected_movies]
        
        # 2. Get the math vectors for those movies and average them to find the user's "taste profile"
        user_profile = np.mean(self.tmdb_latent[selected_indices], axis=0).reshape(1, -1)
        
        # 3. Calculate similarity against ALL 11,500 movies instantly
        similarities = cosine_similarity(user_profile, self.tmdb_latent)[0]
        
        # 4. Sort and filter
        exclude_indices = set(selected_indices)
        if watch_history:
            exclude_indices.update([m["index"] for m in watch_history])

        top_indices = np.argsort(similarities)[::-1]

        final_recs = []
        for idx in top_indices:
            if len(final_recs) >= top_k: break
            if idx in exclude_indices: continue

            row = self.tmdb_df.iloc[idx]

            if language_filter and row.get("original_language") != language_filter: continue

            if genre_filters:
                row_genres = row.get("genres", "")
                if not all(g in str(row_genres) for g in genre_filters): continue

            rec_dict = row.to_dict()
            rec_dict["hybrid_score"] = float(similarities[idx])
            final_recs.append(rec_dict)

        return pd.DataFrame(final_recs), {"info": "Powered by Fast Cosine Similarity"}
