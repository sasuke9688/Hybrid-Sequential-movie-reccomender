# 2. The Recommendation Engine Interface
class RecommendationEngine:
    def __init__(self, model, content_matrix, tmdb_df, mlb):
        self.model = model
        self.model.eval() # Lock weights for inference
        self.content_matrix = content_matrix
        self.tmdb_df = tmdb_df
        self.mlb = mlb
        self.num_items = len(tmdb_df)

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
        """Executes a PyTorch forward pass with extreme RAM optimization."""
        # Setup context
        user_idx = torch.LongTensor([0]) 
        
        # Build sequence from history and selected movies
        seq_list = []
        if watch_history:
            seq_list.extend([movie["index"] for movie in watch_history])
        seq_list.extend([movie["index"] for movie in selected_movies])
        
        # Truncate sequence to max 20, pad with 0 if needed
        seq_list = seq_list[-20:] 
        if len(seq_list) < 20:
            seq_list = [0] * (20 - len(seq_list)) + seq_list
            
        seq_tensor = torch.LongTensor([seq_list])

        # Prepare batch for all items in catalog
        all_item_indices = torch.arange(1, self.num_items + 1)
        user_batch = user_idx.expand(self.num_items)
        seq_batch = seq_tensor.expand(self.num_items, 20)
        
        # Memory constraint: Disable gradient tracking
        with torch.no_grad():
            scores = self.model(user_batch, all_item_indices, self.content_matrix[1:], seq_batch)
        
        # 1. Flatten scores to a 1D NumPy array
        score_array = scores.numpy().flatten()

        # 2. Identify movies to exclude (the ones the user just clicked/watched)
        exclude_indices = set([m["index"] for m in selected_movies])
        if watch_history:
            exclude_indices.update([m["index"] for m in watch_history])

        # 3. Mathematically sort the scores from highest to lowest (No Pandas copying!)
        top_indices = np.argsort(score_array)[::-1]

        # 4. Extract ONLY the winning movies that match the filters
        final_recs = []
        for idx in top_indices:
            if len(final_recs) >= top_k:
                break # Stop searching once we have enough movies
                
            if idx in exclude_indices:
                continue

            # Extract just this single row (Extremely RAM efficient)
            row = self.tmdb_df.iloc[idx]

            # Apply Language Filter
            if language_filter and row.get("original_language") != language_filter:
                continue

            # Apply Genre Filter
            if genre_filters:
                row_genres = row.get("genres", [])
                if not isinstance(row_genres, list):
                    # Handle safely if it's a string
                    row_genres = [g.strip() for g in str(row_genres).split(",")]
                
                # Check if all required genres exist in the movie's genres
                if not all(g in row_genres for g in genre_filters):
                    continue

            # If it survives the filters, add it to our final list
            rec_dict = row.to_dict()
            rec_dict["hybrid_score"] = float(score_array[idx])
            final_recs.append(rec_dict)

        # Return just the small subset of winning movies
        return pd.DataFrame(final_recs), {"info": "Powered by PyTorch (Memory-Optimized)"}
