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
# Import the fixed drug sensitivity module
from drug_sensitivity_enhanced import DrugSensitivityPredictor
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib parameters for better visualization
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

pickle_file = 'preprocessed_data.pkl'
# Load preprocessed data
with open(pickle_file, 'rb') as f:
    data = pickle.load(f)
gene_expr_scaled = data['gene_expr_scaled']
methylation_scaled = data['methylation_scaled']
cna_scaled = data['cna_scaled']
pam50_labels = data['pam50_labels']
survival_times = data['survival_times']
survival_events = data['survival_events']
valid_survival = data['valid_survival']
label_encoder = data['label_encoder']

pickle_file = 'graph_data.pkl'
# Load graph data
with open(pickle_file, 'rb') as f:
    data = pickle.load(f)
# Extract graph data components
graph_data = data['graph_data']
selected_features = data['selected_features']
selected_indices = data['selected_indices']
lasso_model = data['lasso_model']

class MultiModalGAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=8, dropout=0.6, num_layers=3):
        super(MultiModalGAT, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GAT layers
        self.gat_layers = nn.ModuleList()
        
        # First layer
        self.gat_layers.append(
            GATv2Conv(input_dim, hidden_dim, heads=num_heads, dropout=dropout, concat=True)
        )
        
        # Intermediate layers
        for _ in range(num_layers - 2):
            self.gat_layers.append(
                GATv2Conv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout, concat=True)
            )
        
        # Final layer
        self.gat_layers.append(
            GATv2Conv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout, concat=False)
        )
        
        # Task-specific heads (modified for node-level predictions)
        self.pam50_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        self.survival_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # Apply GAT layers
        for i, gat_layer in enumerate(self.gat_layers):
            x = gat_layer(x, edge_index)
            if i < len(self.gat_layers) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Remove global pooling - we want node-level predictions
        # Directly use node embeddings for predictions
        pam50_logits = self.pam50_classifier(x)
        survival_pred = self.survival_predictor(x)
        
        return pam50_logits, survival_pred

class TreatmentRecommendationSystem:
    """Treatment recommendation system based on PAM50 subtypes"""
    
    def __init__(self):
        self.treatment_map = {
            0: {  # Luminal A
                'first_line': ['Tamoxifen', 'Anastrozole', 'Letrozole'],
                'targeted': ['CDK4/6 inhibitors (Palbociclib)', 'mTOR inhibitors'],
                'prognosis': 'Excellent (95.6% 5-year survival)',
                'considerations': 'Hormone-responsive, avoid unnecessary chemotherapy'
            },
            1: {  # Luminal B
                'first_line': ['Hormone therapy + Chemotherapy', 'CDK4/6 inhibitors'],
                'targeted': ['Fulvestrant', 'Ribociclib', 'Abemaciclib'],
                'prognosis': 'Good (91.8% 5-year survival)',
                'considerations': 'Higher proliferation, may need chemotherapy'
            },
            2: {  # HER2-enriched
                'first_line': ['Trastuzumab + Pertuzumab + Chemotherapy'],
                'targeted': ['T-DM1', 'Tucatinib', 'Neratinib'],
                'prognosis': 'Good with treatment (86.5% 5-year survival)',
                'considerations': 'HER2-targeted therapy essential'
            },
            3: {  # Triple-negative/Basal-like
                'first_line': ['Chemotherapy', 'Immunotherapy combinations'],
                'targeted': ['PARP inhibitors (BRCA+)', 'Pembrolizumab', 'Atezolizumab'],
                'prognosis': 'Challenging (78.4% 5-year survival)',
                'considerations': 'Limited targeted options, check BRCA status'
            }
        }
    
    def generate_treatment_recommendations(self, subtype_predictions):
        """Generate evidence-based treatment recommendations"""
        recommendations = []
        for subtype in subtype_predictions:
            if subtype in self.treatment_map:
                recommendations.append(self.treatment_map[subtype])
            else:
                recommendations.append({'error': 'Unknown subtype'})
        return recommendations
    
    def create_treatment_summary_table(self, predictions):
        """Create treatment summary table"""
        unique_subtypes = np.unique(predictions)
        subtype_names = ['Luminal A', 'Luminal B', 'HER2-enriched', 'Triple-negative']
        
        summary_data = []
        for subtype in unique_subtypes:
            count = np.sum(predictions == subtype)
            if subtype < len(subtype_names):
                treatment_info = self.treatment_map[subtype]
                summary_data.append({
                    'Subtype': subtype_names[subtype],
                    'Count': count,
                    'Percentage': f"{(count/len(predictions)*100):.1f}%",
                    'First-line Treatment': ', '.join(treatment_info['first_line'][:2]),
                    'Prognosis': treatment_info['prognosis']
                })
        
        return pd.DataFrame(summary_data)

# Initialize model
input_dim = selected_features.shape[1]
hidden_dim = 256
output_dim = len(label_encoder.classes_)

model = MultiModalGAT(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    output_dim=output_dim,
    num_heads=8,
    dropout=0.6,
    num_layers=3
)

print(f"Model initialized with input_dim={input_dim}, output_dim={output_dim}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

def cox_ph_loss(risk_scores, survival_times, events):
    """Partial likelihood loss for Cox proportional hazards model"""
    # Sort by survival time (descending)
    sorted_indices = torch.argsort(survival_times, descending=True)
    sorted_risk_scores = risk_scores[sorted_indices]
    sorted_events = events[sorted_indices]
    
    hazard_ratios = torch.exp(sorted_risk_scores)
    log_risk = torch.log(torch.cumsum(hazard_ratios, dim=0) + 1e-7)
    
    uncensored_likelihood = sorted_risk_scores - log_risk
    censored_likelihood = uncensored_likelihood * sorted_events
    
    return -torch.sum(censored_likelihood) / (torch.sum(sorted_events) + 1e-7)

def combined_loss(pam50_logits, survival_pred, pam50_labels, survival_times, events, alpha=0.7, beta=0.3):
    """Combined loss for multi-task learning"""
    # PAM50 classification loss
    classification_loss = F.cross_entropy(pam50_logits, pam50_labels)
    
    # Cox proportional hazards loss for survival
    risk_scores = survival_pred.squeeze()
    survival_loss = cox_ph_loss(risk_scores, survival_times, events)
    
    total_loss = alpha * classification_loss + beta * survival_loss
    
    return total_loss, classification_loss, survival_loss

def train_model(model, data, num_epochs=300, lr=0.001, weight_decay=5e-4):
    """Updated training loop for node-level predictions"""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
    
    # Move data to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)
    
    print(f"Training on device: {device}")
    
    # Store training history
    training_history = {
        'total_loss': [], 'cls_loss': [], 'surv_loss': [], 'accuracy': []
    }
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Forward pass
        pam50_logits, survival_pred = model(data.x, data.edge_index, data.edge_attr)
        
        # Calculate loss
        total_loss, cls_loss, surv_loss = combined_loss(
            pam50_logits, survival_pred, data.y, data.survival_time, data.event
        )
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        with torch.no_grad():
            pred_labels = torch.argmax(pam50_logits, dim=1)
            accuracy = (pred_labels == data.y).float().mean().item()
        
        # Store history
        training_history['total_loss'].append(total_loss.item())
        training_history['cls_loss'].append(cls_loss.item())
        training_history['surv_loss'].append(surv_loss.item())
        training_history['accuracy'].append(accuracy)
        
        if epoch % 20 == 0:
            print(f'Epoch {epoch:3d}: Total Loss={total_loss:.4f}, '
                  f'Cls Loss={cls_loss:.4f}, Surv Loss={surv_loss:.4f}, '
                  f'Accuracy={accuracy:.4f}')
        
        scheduler.step(total_loss)
    
    # Plot training curves
    plot_training_curves(training_history)
    
    return model

def plot_training_curves(training_history):
    """Plot training loss and accuracy curves"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(len(training_history['total_loss']))
    
    # Total loss
    ax1.plot(epochs, training_history['total_loss'], 'b-', linewidth=2)
    ax1.set_title('Total Training Loss', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    
    # Classification loss
    ax2.plot(epochs, training_history['cls_loss'], 'r-', linewidth=2)
    ax2.set_title('Classification Loss', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.grid(True, alpha=0.3)
    
    # Survival loss
    ax3.plot(epochs, training_history['surv_loss'], 'g-', linewidth=2)
    ax3.set_title('Survival Loss', fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.grid(True, alpha=0.3)
    
    # Accuracy
    ax4.plot(epochs, training_history['accuracy'], 'm-', linewidth=2)
    ax4.set_title('Training Accuracy', fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Training curves saved to 'training_curves.png'")

# Train the model
print("Starting model training...")
trained_model = train_model(model, graph_data, num_epochs=300, lr=0.001)
print("Training completed!")

def evaluate_model(model, data):
    """Evaluate model performance"""
    model.eval()
    device = next(model.parameters()).device
    data = data.to(device)
    
    with torch.no_grad():
        pam50_logits, survival_pred = model(data.x, data.edge_index, data.edge_attr)
        
        # PAM50 classification metrics
        pred_labels = torch.argmax(pam50_logits, dim=1)
        accuracy = (pred_labels == data.y).float().mean().item()
        
        # Convert to numpy for sklearn metrics
        y_true = data.y.cpu().numpy()
        y_pred = pred_labels.cpu().numpy()
        
        # Survival metrics
        risk_scores = survival_pred.squeeze().cpu().numpy()
        survival_times = data.survival_time.cpu().numpy()
        events = data.event.cpu().numpy()
        
        try:
            c_index = concordance_index(survival_times, -risk_scores, events)
        except:
            c_index = 0.5  # Default if calculation fails
    
    return accuracy, c_index, y_true, y_pred, risk_scores

# Evaluate model
accuracy, c_index, y_true, y_pred, risk_scores = evaluate_model(trained_model, graph_data)

print(f"\nModel Performance:")
print(f"PAM50 Classification Accuracy: {accuracy:.4f}")
print(f"Survival C-index: {c_index:.4f}")

# Detailed classification report
if label_encoder is not None:
    target_names = label_encoder.classes_
else:
    target_names = [f'Subtype_{i}' for i in range(len(np.unique(y_true)))]

print("\nDetailed Classification Report:")
print(classification_report(y_true, y_pred, target_names=target_names))

def create_comprehensive_plots(y_true, y_pred, risk_scores, survival_times, events, selected_features, lasso_model):
    """Create comprehensive visualization suite for ISEF project"""
    
    print("\n" + "="*50)
    print("GENERATING COMPREHENSIVE VISUALIZATIONS")
    print("="*50)
    
    # 1. Basic performance plots
    plot_basic_results(y_true, y_pred, risk_scores, survival_times, events)
    
    # 2. Multi-omics heatmap with subtype annotations
    plot_multi_omics_heatmap(selected_features, y_pred)
    
    # 3. t-SNE visualization of patient similarities
    plot_tsne_subtypes(selected_features, y_pred)
    
    # 4. Enhanced survival analysis
    plot_kaplan_meier_curves(y_pred, survival_times, events)
    
    # 5. LASSO feature importance
    plot_lasso_feature_importance(lasso_model, selected_features)
    
    # 6. Subtype distribution
    plot_subtype_distribution(y_pred)

def plot_basic_results(y_true, y_pred, risk_scores, survival_times, events):
    """Plot basic model performance results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', ax=axes[0,0], cmap='Blues')
    axes[0,0].set_title('PAM50 Classification Confusion Matrix', fontweight='bold')
    axes[0,0].set_xlabel('Predicted')
    axes[0,0].set_ylabel('True')
    
    # Risk score distribution
    axes[0,1].hist(risk_scores, bins=20, alpha=0.7, edgecolor='black', color='skyblue')
    axes[0,1].set_title('Risk Score Distribution', fontweight='bold')
    axes[0,1].set_xlabel('Risk Score')
    axes[0,1].set_ylabel('Frequency')
    
    # Risk stratification
    median_risk = np.median(risk_scores)
    high_risk = risk_scores > median_risk
    low_risk = risk_scores <= median_risk
    
    axes[1,0].scatter(survival_times[high_risk], risk_scores[high_risk], 
                     c='red', alpha=0.6, label='High Risk')
    axes[1,0].scatter(survival_times[low_risk], risk_scores[low_risk], 
                     c='blue', alpha=0.6, label='Low Risk')
    axes[1,0].set_xlabel('Survival Time (months)')
    axes[1,0].set_ylabel('Risk Score')
    axes[1,0].set_title('Risk Stratification', fontweight='bold')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Model performance metrics
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [accuracy, precision, recall, f1]
    
    bars = axes[1,1].bar(metrics, values, color=['blue', 'green', 'orange', 'red'], alpha=0.7)
    axes[1,1].set_title('Model Performance Metrics', fontweight='bold')
    axes[1,1].set_ylabel('Score')
    axes[1,1].set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                      f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('model_performance_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Basic performance results saved to 'model_performance_results.png'")

def plot_multi_omics_heatmap(features, predictions):
    """Create clustered heatmap of selected features by subtype"""
    try:
        # Create subtype-specific feature matrix
        df_plot = pd.DataFrame(features)
        df_plot['Subtype'] = predictions
        
        # Calculate mean expression by subtype
        subtype_means = df_plot.groupby('Subtype').mean()
        
        # Select top varying features for better visualization
        feature_vars = subtype_means.var(axis=0)
        top_features = feature_vars.nlargest(50).index
        
        # Create clustered heatmap
        plt.figure(figsize=(12, 10))
        sns.clustermap(subtype_means[top_features].T, cmap='RdBu_r', center=0, 
                       figsize=(12, 10), cbar_kws={'label': 'Normalized Expression'},
                       row_cluster=True, col_cluster=True)
        plt.savefig('multi_omics_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Multi-omics heatmap saved to 'multi_omics_heatmap.png'")
    except Exception as e:
        print(f"Warning: Could not create multi-omics heatmap: {e}")

def plot_tsne_subtypes(features, predictions):
    """Create t-SNE visualization colored by predicted subtypes"""
    try:
        from sklearn.manifold import TSNE
        
        # Reduce features if too many for t-SNE
        if features.shape[1] > 50:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=50)
            features_reduced = pca.fit_transform(features)
        else:
            features_reduced = features
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)//4))
        tsne_results = tsne.fit_transform(features_reduced)
        
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], 
                             c=predictions, cmap='Set1', alpha=0.7, s=50)
        
        # Add colorbar with subtype labels
        cbar = plt.colorbar(scatter, label='PAM50 Subtype')
        subtype_names = ['Luminal A', 'Luminal B', 'HER2-enriched', 'Triple-negative']
        cbar.set_ticks(range(len(subtype_names)))
        cbar.set_ticklabels(subtype_names)
        
        plt.xlabel('t-SNE Component 1', fontsize=12)
        plt.ylabel('t-SNE Component 2', fontsize=12)
        plt.title('t-SNE Visualization of Patient Similarities by Subtype', 
                 fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.savefig('tsne_subtypes.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ t-SNE visualization saved to 'tsne_subtypes.png'")
    except Exception as e:
        print(f"Warning: Could not create t-SNE plot: {e}")

def plot_kaplan_meier_curves(predictions, survival_times, events):
    """Create Kaplan-Meier survival curves by subtype"""
    try:
        from lifelines import KaplanMeierFitter
        
        plt.figure(figsize=(12, 8))
        kmf = KaplanMeierFitter()
        
        unique_subtypes = np.unique(predictions)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        subtype_names = ['Luminal A', 'Luminal B', 'HER2-enriched', 'Triple-negative']
        
        for i, subtype in enumerate(unique_subtypes):
            if subtype < len(subtype_names):
                mask = predictions == subtype
                if np.sum(mask) > 0:  # Only plot if we have samples
                    kmf.fit(survival_times[mask], events[mask], 
                           label=f'{subtype_names[subtype]} (n={np.sum(mask)})')
                    kmf.plot_survival_function(color=colors[i % len(colors)], linewidth=3)
        
        plt.xlabel('Time (months)', fontsize=12)
        plt.ylabel('Survival Probability', fontsize=12)
        plt.title('Kaplan-Meier Survival Curves by PAM50 Subtype', 
                 fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.xlim(0, None)
        plt.ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig('kaplan_meier_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Kaplan-Meier curves saved to 'kaplan_meier_curves.png'")
    except Exception as e:
        print(f"Warning: Could not create Kaplan-Meier curves: {e}")

def plot_lasso_feature_importance(lasso_model, selected_features):
    """Plot LASSO feature importance"""
    try:
        if hasattr(lasso_model, 'coef_'):
            # Get feature importance
            importance = np.abs(lasso_model.coef_)
            
            # Get top 20 features
            top_20_idx = np.argsort(importance)[-20:]
            top_20_importance = importance[top_20_idx]
            
            plt.figure(figsize=(10, 8))
            plt.barh(range(20), top_20_importance, color='steelblue', alpha=0.7)
            plt.xlabel('Absolute LASSO Coefficient', fontsize=12)
            plt.ylabel('Features', fontsize=12)
            plt.title('Top 20 Most Important Features (LASSO)', fontsize=14, fontweight='bold')
            plt.yticks(range(20), [f'Feature_{i}' for i in top_20_idx])
            plt.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            plt.savefig('lasso_feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ LASSO feature importance saved to 'lasso_feature_importance.png'")
    except Exception as e:
        print(f"Warning: Could not create feature importance plot: {e}")

def plot_subtype_distribution(predictions):
    """Plot distribution of predicted subtypes with proper error handling"""
    subtype_names = ['Luminal A', 'Luminal B', 'HER2-enriched', 'Triple-negative']
    unique_subtypes, counts = np.unique(predictions, return_counts=True)
    
    # Filter out any subtypes that don't have corresponding names
    valid_subtypes = []
    valid_counts = []
    valid_names = []
    
    for i, subtype in enumerate(unique_subtypes):
        if subtype < len(subtype_names):  # Only include valid indices
            valid_subtypes.append(subtype)
            valid_counts.append(counts[i])
            valid_names.append(subtype_names[subtype])
    
    if not valid_subtypes:
        print("No valid subtypes found for distribution plot")
        return
    
    plt.figure(figsize=(10, 6))
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    
    # Use valid names directly instead of indexing
    bars = plt.bar(valid_names, valid_counts, 
                   color=[colors[i] for i in valid_subtypes], 
                   alpha=0.8, edgecolor='black')
    
    # Add count labels on bars
    for bar, count in zip(bars, valid_counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Cancer Subtype', fontsize=12)
    plt.ylabel('Number of Patients', fontsize=12)
    plt.title('Distribution of Predicted Cancer Subtypes', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('subtype_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Subtype distribution saved to 'subtype_distribution.png'")


# Generate all visualizations
create_comprehensive_plots(y_true, y_pred, risk_scores,
                          graph_data.survival_time.cpu().numpy(),
                          graph_data.event.cpu().numpy(),
                          selected_features, lasso_model)

# Treatment recommendation analysis
print("\n" + "="*50)
print("TREATMENT RECOMMENDATION ANALYSIS")
print("="*50)

# Initialize treatment system
treatment_system = TreatmentRecommendationSystem()

# Generate recommendations
recommendations = treatment_system.generate_treatment_recommendations(y_pred)

# Create and display treatment summary
treatment_summary = treatment_system.create_treatment_summary_table(y_pred)
print("\nTreatment Summary by Subtype:")
print(treatment_summary.to_string(index=False))

# Save treatment summary
treatment_summary.to_csv('treatment_summary.csv', index=False)
print("✓ Treatment summary saved to 'treatment_summary.csv'")

# Drug sensitivity analysis (FIXED)
print("\n" + "="*50)
print("DRUG SENSITIVITY ANALYSIS")
print("="*50)

drug_predictor = DrugSensitivityPredictor()
drug_predictor.plot_drug_sensitivity_heatmap(y_pred)

# Save the trained model and results
print("\n" + "="*50)
print("SAVING RESULTS")
print("="*50)

# Save model
pickle_file = 'trained_model.pkl'
with open(pickle_file, 'wb') as f:
    pickle.dump({
        'model_state_dict': trained_model.state_dict(),
        'model': trained_model,
        'accuracy': accuracy,
        'c_index': c_index,
        'predictions': y_pred,
        'true_labels': y_true,
        'risk_scores': risk_scores,
        'report': classification_report(y_true, y_pred, target_names=target_names)
    }, f)
print("✓ Trained model saved to 'trained_model.pkl'")

# Create results summary
results_summary = {
    'Model Performance': {
        'PAM50 Classification Accuracy': f"{accuracy:.4f}",
        'Survival C-index': f"{c_index:.4f}"
    },
    'Generated Files': [
        'training_curves.png',
        'model_performance_results.png',
        'multi_omics_heatmap.png',
        'tsne_subtypes.png',
        'kaplan_meier_curves.png',
        'lasso_feature_importance.png',
        'subtype_distribution.png',
        'drug_sensitivity_heatmap.png',
        'treatment_summary.csv',
        'trained_model.pkl'
    ]
}

# Save results summary
with open('results_summary.txt', 'w') as f:
    f.write("MULTI-OMICS CANCER SUBTYPE PREDICTION - RESULTS SUMMARY\n")
    f.write("="*60 + "\n\n")
    f.write(f"PAM50 Classification Accuracy: {accuracy:.4f}\n")
    f.write(f"Survival C-index: {c_index:.4f}\n\n")
    f.write("Generated Visualizations:\n")
    for file in results_summary['Generated Files']:
        f.write(f"  - {file}\n")
    f.write(f"\nClassification Report:\n")
    f.write(classification_report(y_true, y_pred, target_names=target_names))

print("✓ Results summary saved to 'results_summary.txt'")

print("\n" + "="*60)
print("🎉 ANALYSIS COMPLETED SUCCESSFULLY! 🎉")
print("="*60)
print(f"✓ Model Accuracy: {accuracy:.4f}")
print(f"✓ Survival C-index: {c_index:.4f}")
print(f"✓ Generated {len(results_summary['Generated Files'])} output files")
print("✓ All visualizations and results saved successfully!")
print("="*60)
