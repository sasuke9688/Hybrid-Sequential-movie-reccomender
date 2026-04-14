import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicHybridRecommender(nn.Module):
    def __init__(self, num_users, num_items, content_feature_dim, latent_dim=100):
        super(DynamicHybridRecommender, self).__init__()
        
        # 1. Collaborative Filtering Module (Replaces TruncatedSVD)
        self.user_emb = nn.Embedding(num_users, latent_dim)
        self.item_emb = nn.Embedding(num_items, latent_dim)
        
        # 2. Content-Based Module (Replaces Ridge Regression)
        # Maps raw TMDB content vectors (TF-IDF/Genres) to the latent space
        self.content_mlp = nn.Sequential(
            nn.Linear(content_feature_dim, latent_dim),
            nn.ReLU(),
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim)
        )
        
        # 3. Sequential Module
        # Captures temporal intent from chronological user interaction history
        self.seq_encoder = nn.GRU(latent_dim, latent_dim, batch_first=True)
        
        # 4. Context-Aware Dynamic Gating Network
        # Learns to assign weights to CF, CBF, and Seq scores dynamically per interaction
        self.gating_network = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, 3) # Outputs exactly 3 weights
        )

    def forward(self, user_idx, item_idx, item_content_vector, user_history_seq):
        """
        user_idx: Tensor of target user IDs
        item_idx: Tensor of target item IDs
        item_content_vector: Tensor of TMDB item features (e.g., TF-IDF + genres)
        user_history_seq: Tensor of chronological item IDs the user previously engaged with
        """
        # A. Compute Collaborative Signal
        u_cf = self.user_emb(user_idx)
        v_cf = self.item_emb(item_idx)
        score_cf = (u_cf * v_cf).sum(dim=1)
        
        # B. Compute Content-Based Signal
        v_cbf = self.content_mlp(item_content_vector)
        score_cbf = (u_cf * v_cbf).sum(dim=1)
        
        # C. Compute Sequential Signal
        history_embs = self.item_emb(user_history_seq)
        _, h_n = self.seq_encoder(history_embs)
        h_t = h_n[-1]  # Extract the final hidden state (current temporal intent)
        score_seq = (h_t * v_cf).sum(dim=1)
        
        # D. Dynamic Fusion (Attention Mechanism)
        # The weights are generated based purely on the user's current sequential state
        gate_logits = self.gating_network(h_t)
        gate_weights = F.softmax(gate_logits, dim=1)
        
        w_cf  = gate_weights[:, 0]
        w_cbf = gate_weights[:, 1]
        w_seq = gate_weights[:, 2]
        
        # Final output is the dynamically weighted combination of all three models
        final_score = (w_cf * score_cf) + (w_cbf * score_cbf) + (w_seq * score_seq)
        
        return torch.sigmoid(final_score)
