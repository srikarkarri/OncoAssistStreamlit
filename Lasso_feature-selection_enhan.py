import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv, global_mean_pool
from torch_geometric.data import Data, DataLoader
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LassoCV, Lasso
from lifelines.utils import concordance_index
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.spatial.distance as distance
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

pickle_file = 'preprocessed_data.pkl'
# Load preprocessed data
with open(pickle_file, 'rb') as f:
    data = pickle.load(f)
# Extract data components
gene_expr_scaled = data['gene_expr_scaled']
methylation_scaled = data['methylation_scaled']
cna_scaled = data['cna_scaled']
pam50_labels = data['pam50_labels']
survival_times = data['survival_times']
survival_events = data['survival_events']
valid_survival = data['valid_survival']
label_encoder = data['label_encoder']

def lasso_feature_selection(X, y, alpha=None, max_features=1000):
    """
    Apply LASSO regression for feature selection
    """
    print(f"Applying LASSO feature selection to {X.shape[1]} features...")
    
    if alpha is None:
        lasso = LassoCV(cv=5, random_state=42, max_iter=2000, n_alphas=50)
    else:
        lasso = Lasso(alpha=alpha, max_iter=2000)
    
    lasso.fit(X, y)
    
    # Select features with non-zero coefficients
    selected_features = np.where(lasso.coef_ != 0)[0]
    
    # If too many features selected, take top ones by coefficient magnitude
    if len(selected_features) > max_features:
        coef_abs = np.abs(lasso.coef_[selected_features])
        top_indices = np.argsort(coef_abs)[::-1][:max_features]
        selected_features = selected_features[top_indices]
    
    print(f"Selected {len(selected_features)} features using LASSO")
    
    return X[:, selected_features], selected_features, lasso

# Apply LASSO feature selection
print("Applying LASSO feature selection...")

# Combine all molecular data
combined_features = np.concatenate([
    gene_expr_scaled.values,
    methylation_scaled.values,
    cna_scaled.values
], axis=1)

print(f"Combined features shape: {combined_features.shape}")

# Apply LASSO feature selection for PAM50 classification
selected_features, selected_indices, lasso_model = lasso_feature_selection(
    combined_features, pam50_labels, max_features=500
)

print(f"Final selected features shape: {selected_features.shape}")
def construct_patient_similarity_graph(features, similarity_metric='cosine', k_neighbors=10):
    """
    Construct patient similarity graph based on multi-omics features
    """
    from sklearn.metrics.pairwise import cosine_similarity
    from scipy.spatial import distance
    import numpy as np
    print(f"Constructing patient similarity graph using {similarity_metric} similarity...")
    
    if similarity_metric == 'cosine':
        similarity_matrix = cosine_similarity(features)
    elif similarity_metric == 'euclidean':
        dist_matrix = distance.pdist(features, metric='euclidean')
        dist_matrix = distance.squareform(dist_matrix)
        similarity_matrix = 1 / (1 + dist_matrix)
    elif similarity_metric == 'correlation':
        similarity_matrix = np.corrcoef(features)
        similarity_matrix = np.nan_to_num(similarity_matrix, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Keep only top-k neighbors for each patient
    n_samples = similarity_matrix.shape[0]
    edge_index = []
    edge_attr = []
    
    for i in range(n_samples):
        neighbors = np.argsort(similarity_matrix[i])[::-1][1:k_neighbors+1]
        for j in neighbors:
            edge_index.append([i, j])
            edge_attr.append(similarity_matrix[i][j])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
    print(f"Created graph with {edge_index.shape[1]} edges")
    
    return edge_index, edge_attr

def create_pytorch_geometric_data(features, labels, survival_times, survival_events, valid_survival):
    """
    Creates PyTorch Geometric Data object with clinical-grade data validation
    
    Args:
        features: NumPy array of molecular features (samples x features)
        labels: Array of PAM50 subtype labels
        survival_times: Array of survival times in months
        survival_events: Array of survival status indicators
        valid_survival: Boolean mask indicating valid samples
    
    Returns:
        PyG Data object with multi-modal cancer data
    """
    
    # 1. Filter valid samples
    features_valid = features[valid_survival]
    labels_valid = labels[valid_survival]
    
    # 2. Process survival times with clinical validation
    def clean_survival_time(t):
        """Convert various time formats to float"""
        try:
            # Handle comma decimals and string representations
            return float(str(t).replace(',', '.').strip())
        except:
            return np.nan
    
    # Convert and clean survival times
    survival_times_clean = [clean_survival_time(t) for t in survival_times[valid_survival]]
    survival_times_series = pd.Series(survival_times_clean)
    
    # Impute missing values with cohort median (better than 0 for clinical validity)
    median_survival = survival_times_series.median()
    survival_times_valid = survival_times_series.fillna(median_survival).to_numpy()
    
    # 3. Process survival events with clinical encoding
    def map_survival_event(event):
        """Convert clinical event codes to binary labels"""
        if isinstance(event, str):
            if event.startswith('1:') or 'DECEASED' in event.upper():
                return 1
            elif event.startswith('0:') or 'LIVING' in event.upper():
                return 0
        try:
            return int(float(event))  # Handle numeric strings
        except:
            return 0  # Default to censored if unknown format
    
    survival_events_valid = np.array(
        [map_survival_event(e) for e in survival_events[valid_survival]],
        dtype=int
    )
    
    # 4. Build patient similarity graph
    edge_index, edge_attr = construct_patient_similarity_graph(features_valid)
    
    # 5. Create PyG Data object with validation
    return Data(
        x=torch.tensor(features_valid, dtype=torch.float32),
        y=torch.tensor(labels_valid, dtype=torch.long),
        survival_time=torch.tensor(survival_times_valid, dtype=torch.float32),
        event=torch.tensor(survival_events_valid, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=edge_attr  # Added edge attributes
    )

def plot_lasso_regularization_path(lasso_model, feature_names=None):
    """Plot LASSO regularization path"""
    if hasattr(lasso_model, 'alphas_'):
        alphas = lasso_model.alphas_
        coefs = lasso_model.path(combined_features, pam50_labels, alphas=alphas)[1]
        
        plt.figure(figsize=(12, 8))
        plt.plot(alphas, coefs.T, linewidth=0.5)
        plt.xscale('log')
        plt.xlabel('Alpha (Regularization Parameter)')
        plt.ylabel('Coefficient Value')
        plt.title('LASSO Regularization Path')
        plt.axvline(lasso_model.alpha_, color='red', linestyle='--', 
                   label=f'Selected Alpha: {lasso_model.alpha_:.4f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('lasso_regularization_path.png', dpi=300, bbox_inches='tight')
        plt.close()

# Add this call before saving the graph data
plot_lasso_regularization_path(lasso_model)

# Create graph data
graph_data = create_pytorch_geometric_data(
    selected_features, pam50_labels, survival_times, survival_events, valid_survival
)

pickle_file = 'graph_data.pkl'
# Save graph data to file
pickle.dump({
    'graph_data': graph_data,
    'selected_features': selected_features,
    'selected_indices': selected_indices,
    'lasso_model': lasso_model
}, open('graph_data.pkl', 'wb'))

print(f"Graph data created with {graph_data.x.shape[0]} nodes and {graph_data.edge_index.shape[1]} edges")
