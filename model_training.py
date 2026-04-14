"""
Dynamic Hybrid Sequential Recommendation System
PyTorch Training Pipeline for TMDB Dataset
"""

import os
import ast
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer

# ==========================================
# 1. Configuration
# ==========================================
INPUT_FILE = "golden_tmdb_11k.csv"
INTERACTIONS_FILE = "user_interactions.csv" # Replace with your actual user logs
MODEL_DIR = "models"
LATENT_DIM = 100
MAX_SEQ_LENGTH = 20
BATCH_SIZE = 256
EPOCHS = 5
LEARNING_RATE = 0.001

# ==========================================
# 2. PyTorch Architecture
# ==========================================
class DynamicHybridRecommender(nn.Module):
    def __init__(self, num_users, num_items, content_feature_dim, latent_dim=100):
        super(DynamicHybridRecommender, self).__init__()
        
        # Collaborative Module
        self.user_emb = nn.Embedding(num_users, latent_dim)
        self.item_emb = nn.Embedding(num_items, latent_dim, padding_idx=0)
        
        # Content Module
        self.content_mlp = nn.Sequential(
            nn.Linear(content_feature_dim, latent_dim),
            nn.ReLU(),
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim)
        )
        
        # Sequential Module
        self.seq_encoder = nn.GRU(latent_dim, latent_dim, batch_first=True)
        
        # Dynamic Gating Network
        self.gating_network = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, 3)
        )

    def forward(self, user_idx, item_idx, item_content_vector, user_history_seq):
        # A. Collaborative Signal
        u_cf = self.user_emb(user_idx)
        v_cf = self.item_emb(item_idx)
        score_cf = (u_cf * v_cf).sum(dim=1)
        
        # B. Content Signal
        v_cbf = self.content_mlp(item_content_vector)
        score_cbf = (u_cf * v_cbf).sum(dim=1)
        
        # C. Sequential Signal
        history_embs = self.item_emb(user_history_seq)
        _, h_n = self.seq_encoder(history_embs)
        h_t = h_n[-1] 
        score_seq = (h_t * v_cf).sum(dim=1)
        
        # D. Dynamic Gating
        gate_logits = self.gating_network(h_t)
        gate_weights = F.softmax(gate_logits, dim=1)
        
        w_cf  = gate_weights[:, 0]
        w_cbf = gate_weights[:, 1]
        w_seq = gate_weights[:, 2]
        
        # Fusion
        final_score = (w_cf * score_cf) + (w_cbf * score_cbf) + (w_seq * score_seq)
        return torch.sigmoid(final_score)

# ==========================================
# 3. PyTorch Dataset
# ==========================================
class SequentialRecommendationDataset(Dataset):
    def __init__(self, interaction_df, item_content_matrix, max_seq_length=20):
        self.users = interaction_df['user_idx'].values
        self.items = interaction_df['item_idx'].values
        self.labels = interaction_df['label'].values 
        self.item_content_matrix = torch.FloatTensor(item_content_matrix)
        self.max_seq_length = max_seq_length
        
        print("Building user sequential histories...")
        self.user_histories = {}
        for uid, iid in zip(self.users, self.items):
            if uid not in self.user_histories:
                self.user_histories[uid] = []
            self.user_histories[uid].append(iid)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        item = self.items[idx]
        label = self.labels[idx]
        content_vec = self.item_content_matrix[item]
        
        full_history = self.user_histories[user]
        target_idx_in_history = full_history.index(item) if item in full_history else len(full_history)
        history = full_history[:target_idx_in_history]
        
        if len(history) >= self.max_seq_length:
            seq = history[-self.max_seq_length:]
        else:
            seq = [0] * (self.max_seq_length - len(history)) + history
            
        return (
            torch.LongTensor([user])[0],
            torch.LongTensor([item])[0],
            content_vec,
            torch.LongTensor(seq),
            torch.FloatTensor([label])[0]
        )

# ==========================================
# 4. Data Preprocessing & Training Logic
# ==========================================
def safe_parse_genres(genre_string):
    if pd.isna(genre_string): return []
    if isinstance(genre_string, list): return genre_string
    try:
        parsed = ast.literal_eval(genre_string)
        if isinstance(parsed, list): return parsed
    except (ValueError, SyntaxError):
        pass
    return [g.strip() for g in str(genre_string).split(',') if g.strip()]

def extract_content_features(df):
    print("Extracting NLP and Genre features...")
    df['genres'] = df['genres'].apply(safe_parse_genres)
    df['overview'] = df['overview'].fillna("")
    df['title'] = df['title'].fillna("")
    
    mlb = MultiLabelBinarizer()
    genre_matrix = mlb.fit_transform(df['genres'])
    
    text_corpus = df['title'] + " " + df['genres'].apply(lambda x: " ".join(x)) + " " + df['overview']
    tfidf = TfidfVectorizer(stop_words='english', max_features=3000)
    tfidf_matrix = tfidf.fit_transform(text_corpus).toarray()
    
    # Prepend a zero-vector for padding_idx = 0
    content_matrix = np.hstack((tfidf_matrix, genre_matrix))
    padding_row = np.zeros((1, content_matrix.shape[1]))
    final_content_matrix = np.vstack((padding_row, content_matrix))
    
    return final_content_matrix, mlb

def train_dynamic_model(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing training on hardware: {device}")
    model.to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for batch_idx, (users, items, content_vecs, seqs, labels) in enumerate(dataloader):
            users, items = users.to(device), items.to(device)
            content_vecs, seqs = content_vecs.to(device), seqs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            predictions = model(users, items, content_vecs, seqs)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS} | Batch {batch_idx} | Loss: {loss.item():.4f}")
                
        print(f"Epoch {epoch+1} Completed | Average Loss: {total_loss/len(dataloader):.4f}")
    return model

def export_artifacts(model, content_matrix, df, mlb):
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Save PyTorch Tensors
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "dynamic_hybrid_model.pth"))
    torch.save(torch.FloatTensor(content_matrix), os.path.join(MODEL_DIR, "item_content_matrix.pt"))
    
    # Save Scikit/Pandas Artifacts
    joblib.dump(df, os.path.join(MODEL_DIR, "tmdb_dataset.pkl"))
    joblib.dump(mlb, os.path.join(MODEL_DIR, "mlb.pkl"))
    print("Artifacts successfully exported to production directory.")

# ==========================================
# 5. Pipeline Execution
# ==========================================
def run_training_pipeline():
    print("=" * 60)
    print("INITIALIZING DYNAMIC HYBRID TRAINING PIPELINE")
    print("=" * 60)

    # 1. Load TMDB Catalog
    tmdb_df = pd.read_csv(INPUT_FILE)
    
    # 2. Process Content Features
    content_matrix, mlb = extract_content_features(tmdb_df)
    
    # 3. Load or Generate Interactions
    # Note: Replace this block with your actual user interaction loading logic
    if os.path.exists(INTERACTIONS_FILE):
        interactions = pd.read_csv(INTERACTIONS_FILE)
    else:
        print("WARNING: user_interactions.csv not found. Generating mock chronological data for compilation testing.")
        num_mock_users = 1000
        num_mock_items = len(tmdb_df)
        interactions = pd.DataFrame({
            'user_idx': np.random.randint(1, num_mock_users, 10000),
            # item_idx shifted by 1 because 0 is reserved for sequence padding
            'item_idx': np.random.randint(1, num_mock_items + 1, 10000), 
            'label': np.random.choice([0.0, 1.0], 10000, p=[0.2, 0.8]),
            'timestamp': np.sort(np.random.randint(1600000000, 1700000000, 10000))
        }).sort_values('timestamp')

    num_users = interactions['user_idx'].max() + 1
    num_items = len(tmdb_df) + 1 # +1 for padding index 0
    content_dim = content_matrix.shape[1]

    # 4. Initialize Data Loaders
    print("Configuring DataLoader matrices...")
    dataset = SequentialRecommendationDataset(interactions, content_matrix, max_seq_length=MAX_SEQ_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 5. Initialize Model
    print("Instantiating multi-gate architecture...")
    model = DynamicHybridRecommender(
        num_users=num_users, 
        num_items=num_items, 
        content_feature_dim=content_dim, 
        latent_dim=LATENT_DIM
    )

    # 6. Train
    model = train_dynamic_model(model, dataloader)

    # 7. Export
    export_artifacts(model, content_matrix, tmdb_df, mlb)
    
    print("=" * 60)
    print("PIPELINE COMPLETE.")
    print("=" * 60)

if __name__ == "__main__":
    run_training_pipeline()
