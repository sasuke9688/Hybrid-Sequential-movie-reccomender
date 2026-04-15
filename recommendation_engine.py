import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

# 1. The PyTorch Architecture
class DynamicHybridRecommender(nn.Module):
    def __init__(self, num_users, num_items, content_feature_dim, latent_dim=100):
        super(DynamicHybridRecommender, self).__init__()
        self.user_emb = nn.Embedding(num_users, latent_dim)
        self.item_emb = nn.Embedding(num_items, latent_dim, padding_idx=0)
        self.content_mlp = nn.Sequential(
            nn.Linear(content_feature_dim, latent_dim),
            nn.ReLU(),
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim)
        )
        self.seq_encoder = nn.GRU(latent_dim, latent_dim, batch_first=True)
        self.gating_network = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, 3)
        )

    def forward(self, user_idx, item_idx, item_content_vector, user_history_seq):
        u_cf = self.user_emb(user_idx)
        v_cf = self.item_emb(item_idx)
        score_cf = (u_cf * v_cf).sum(dim=1)
        
        v_cbf = self.content_mlp(item_content_vector)
        score_cbf = (u_cf * v_cbf).sum(dim=1)
        
        history_embs = self.item_emb(user_history_seq)
        _, h_n = self.seq_encoder(history_embs)
        h_t = h_n[-1] 
        score_seq = (h_t * v_cf).sum(dim=1)
        
        gate_logits = self.gating_network(h_t)
        gate_weights = F.softmax(gate_logits, dim=1)
        
        w_cf  = gate_weights[:, 0]
        w_cbf = gate_weights[:, 1]
        w_seq = gate_weights[:, 2]
        
        final_score = (w_cf * score_cf) + (w_cbf * score_cbf) + (w_seq * score_seq)
        return torch.sigmoid(final_score)

# 2. The Recommendation Engine Interface (Memory-Safe Version)
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
        user_idx = torch.LongTensor([0]) 
        
        seq_list = []
        if watch_history:
            seq_list.extend([movie["index"] for movie in watch_history])
        seq_list.extend([movie["index"] for movie in selected_movies])
        
        seq_list = seq_list[-20:] 
        if len(seq_list) < 20:
            seq_list = [0] * (20 - len(seq_list)) + seq_list
            
        seq_tensor = torch.LongTensor([seq_list])

        all_item_indices = torch.arange(1, self.num_items + 1)
        user_batch = user_idx.expand(self.num_items)
        seq_batch = seq_tensor.expand(self.num_items, 20)
        
        with torch.no_grad():
            scores = self.model(user_batch, all_item_indices, self.content_matrix[1:], seq_batch)
        
        score_array = scores.numpy().flatten()

        exclude_indices = set([m["index"] for m in selected_movies])
        if watch_history:
            exclude_indices.update([m["index"] for m in watch_history])

        top_indices = np.argsort(score_array)[::-1]

        final_recs = []
        for idx in top_indices:
            if len(final_recs) >= top_k:
                break 
                
            if idx in exclude_indices:
                continue

            row = self.tmdb_df.iloc[idx]

            if language_filter and row.get("original_language") != language_filter:
                continue

            if genre_filters:
                row_genres = row.get("genres", [])
                if not isinstance(row_genres, list):
                    row_genres = [g.strip() for g in str(row_genres).split(",")]
                if not all(g in row_genres for g in genre_filters):
                    continue

            rec_dict = row.to_dict()
            rec_dict["hybrid_score"] = float(score_array[idx])
            final_recs.append(rec_dict)

        return pd.DataFrame(final_recs), {"info": "Powered by PyTorch (Memory-Optimized)"}
