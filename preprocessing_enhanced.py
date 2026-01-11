import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Paths
clinical_path = "/Users/srikarkarri/Precision_Oncology_Final/data_clinical_patient-2.txt"
mutation_path = "/Users/srikarkarri/Precision_Oncology_Final/data_mutations-2.txt"
expression_path = "/Users/srikarkarri/Precision_Oncology_Final/data_mrna_illumina_microarray 2.txt"
methylation_path = "/Users/srikarkarri/Precision_Oncology_Final/data_methylation_promoters_rrbs.txt"
cna_path = "/Users/srikarkarri/Precision_Oncology_Final/data_cna 2.txt"
gene_panel_path = "/Users/srikarkarri/Precision_Oncology_Final/data_gene_panel_matrix 2.txt"

# Install required libraries for Kaggle/Google Colab
# !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# !pip install torch-geometric
# !pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
# !pip install scikit-learn pandas numpy matplotlib seaborn
# !pip install lifelines  # For survival analysis
# !pip install scipy

# Import essential modules
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

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

print("Environment setup complete!")

def load_and_preprocess_data():
    """
    Load and preprocess your specific multi-omics data files
    """
    print("Loading data files...")

    # Load gene expression data (microarray)
    gene_expr = pd.read_csv(expression_path, sep='\t', index_col=0)
    print(f"Gene expression shape: {gene_expr.shape}")

    # Load DNA methylation data
    methylation = pd.read_csv(methylation_path, sep='\t', index_col=0)
    print(f"Methylation data shape: {methylation.shape}")

    # Load copy number alteration data
    cna_data = pd.read_csv(cna_path, sep='\t', index_col=0)
    print(f"CNA data shape: {cna_data.shape}")

    # Load clinical data
    clinical_data = pd.read_csv(clinical_path, sep='\t', index_col=0)
    print(f"Clinical data shape: {clinical_data.shape}")

    # Load mutations data (optional for additional features)
    try:
        mutations = pd.read_csv(mutation_path, sep='\t')
        print(f"Mutations data shape: {mutations.shape}")
    except:
        print("Mutations file not found or couldn't be loaded - proceeding without it")
        mutations = None

    # Load gene panel matrix (for feature selection)
    try:
        gene_panel = pd.read_csv(gene_panel_path, sep='\t', index_col=0)
        print(f"Gene panel matrix shape: {gene_panel.shape}")
    except:
        print("Gene panel matrix not found - proceeding without it")
        gene_panel = None

    return gene_expr, methylation, cna_data, clinical_data, mutations, gene_panel

def preprocess_molecular_data(gene_expr, methylation, cna_data, clinical_data):
    """
    Preprocess and align molecular data with clinical information
    """
    print("Preprocessing molecular data...")

    # Transpose molecular data to have samples as rows (if needed)
    if gene_expr.shape[0] > gene_expr.shape[1]:  # More genes than samples
        gene_expr = gene_expr.T
    if methylation.shape[0] > methylation.shape[1]:  # More CpG sites than samples
        methylation = methylation.T
    if cna_data.shape[0] > cna_data.shape[1]:  # More genes than samples
        cna_data = cna_data.T

    print(f"After transpose - Gene expr: {gene_expr.shape}, Methylation: {methylation.shape}, CNA: {cna_data.shape}")

    # Find common samples across all data types
    common_samples = list(set(gene_expr.index) & set(methylation.index) & set(cna_data.index) & set(clinical_data.index))

# --- NEW: NC CLASS FILTERING ---
    # Identify PAM50 column
    pam50_columns = [col for col in clinical_data.columns
                    if 'pam50' in col.lower() or 'subtype' in col.lower()]

    if not pam50_columns:
        raise ValueError("No PAM50 subtype column found in clinical data")

    pam50_col = pam50_columns[0]

    # Filter out NC samples
    nc_mask = (clinical_data.loc[common_samples, pam50_col] != 'NC').values
    common_samples = np.array(common_samples)[nc_mask].tolist()
    print(f"After NC removal - Remaining samples: {len(common_samples)}")


    if len(common_samples) < 50:
        print("Warning: Very few common samples found. Check sample ID formats.")
        # Try alternative matching strategies
        print("Sample ID examples:")
        print("Gene expr:", gene_expr.index[:5].tolist())
        print("Methylation:", methylation.index[:5].tolist())
        print("CNA:", cna_data.index[:5].tolist())
        print("Clinical:", clinical_data.index[:5].tolist())

    # Subset data to common samples
    gene_expr_aligned = gene_expr.loc[common_samples]
    methylation_aligned = methylation.loc[common_samples]
    cna_aligned = cna_data.loc[common_samples]
    clinical_aligned = clinical_data.loc[common_samples]

    # Handle missing values
    gene_expr_aligned = gene_expr_aligned.fillna(gene_expr_aligned.mean())
    methylation_aligned = methylation_aligned.fillna(methylation_aligned.mean())
    cna_aligned = cna_aligned.fillna(0)  # CNA often uses 0 for no change

    # Remove features with too many missing values or low variance
    gene_expr_aligned = remove_low_variance_features(gene_expr_aligned, variance_threshold=0.01)
    methylation_aligned = remove_low_variance_features(methylation_aligned, variance_threshold=0.01)
    cna_aligned = remove_low_variance_features(cna_aligned, variance_threshold=0.01)

    print(f"After preprocessing - Gene expr: {gene_expr_aligned.shape}, Methylation: {methylation_aligned.shape}, CNA: {cna_aligned.shape}")

    # Standardize features
    scaler_ge = StandardScaler()
    scaler_meth = StandardScaler()
    scaler_cna = StandardScaler()

    gene_expr_scaled = pd.DataFrame(
        scaler_ge.fit_transform(gene_expr_aligned),
        index=gene_expr_aligned.index,
        columns=gene_expr_aligned.columns
    )

    methylation_scaled = pd.DataFrame(
        scaler_meth.fit_transform(methylation_aligned),
        index=methylation_aligned.index,
        columns=methylation_aligned.columns
    )

    cna_scaled = pd.DataFrame(
        scaler_cna.fit_transform(cna_aligned),
        index=cna_aligned.index,
        columns=cna_aligned.columns
    )

    return gene_expr_scaled, methylation_scaled, cna_scaled, clinical_aligned

def remove_low_variance_features(df, variance_threshold=0.01):
    """
    Remove features with low variance
    """
    variances = df.var()
    high_var_features = variances[variances > variance_threshold].index
    return df[high_var_features]

def prepare_labels_and_survival(clinical_data):
    """
    Prepare PAM50 labels and survival data from clinical information
    """
    print("Preparing labels and survival data...")

    # Display available clinical columns
    print("Available clinical columns:")
    print(clinical_data.columns.tolist())

    # Look for PAM50 subtype column (common names)
    pam50_columns = [col for col in clinical_data.columns if 'pam50' in col.lower() or 'subtype' in col.lower()]

    if pam50_columns:
        pam50_col = pam50_columns[0]
        print(f"Found PAM50 column: {pam50_col}")
        pam50_labels = clinical_data[pam50_col].values
        nc_mask = (pam50_labels !='NC')
        print(f"Removing {np.sum(~nc_mask)} NC samples")

        # Apply mask to all relevant data
        pam50_labels_filtered = pam50_labels[nc_mask]
        clinical_data_filtered = clinical_data[nc_mask]


        # Encode PAM50 labels
        label_encoder = LabelEncoder().fit(pam50_labels_filtered)
        pam50_encoded = label_encoder.transform(pam50_labels_filtered)
        print(f"PAM50 classes after NC removal: {label_encoder.classes_}")
        print(f"PAM50 distribution: {np.bincount(pam50_encoded)}")
    else:
        print("No PAM50 column found. Creating dummy labels for demonstration.")
        # Create dummy PAM50 labels for demonstration
        pam50_encoded = np.random.randint(0, 5, size=len(clinical_data))
        label_encoder = None

    # Look for survival columns (common names)
    os_time_cols = [col for col in clinical_data.columns if 'overall' in col.lower() and ('month' in col.lower() or 'day' in col.lower() or 'time' in col.lower())]
    os_status_cols = [col for col in clinical_data.columns if 'overall' in col.lower() and ('status' in col.lower() or 'event' in col.lower())]

    if os_time_cols and os_status_cols:
        os_time_col = os_time_cols[0]
        os_status_col = os_status_cols[0]
        print(f"Found survival columns: {os_time_col}, {os_status_col}")

        survival_times = clinical_data[os_time_col].values[nc_mask]
        survival_events = clinical_data[os_status_col].values[nc_mask]

        # Handle missing survival data
        valid_survival = ~(pd.isna(survival_times) | pd.isna(survival_events))
        print(f"Valid survival data: {np.sum(valid_survival)}/{len(survival_times)}")

    else:
        print("No survival columns found. Creating dummy survival data.")
        # Create dummy survival data
        survival_times = np.random.exponential(24, size=len(clinical_data))  # months
        survival_events = np.random.binomial(1, 0.3, size=len(clinical_data))  # 30% events
        valid_survival = np.ones(len(clinical_data), dtype=bool)

    return pam50_encoded, survival_times, survival_events, valid_survival, label_encoder

# Load and preprocess data
gene_expr, methylation, cna_data, clinical_data, mutations, gene_panel = load_and_preprocess_data()
gene_expr_scaled, methylation_scaled, cna_scaled, clinical_aligned = preprocess_molecular_data(
    gene_expr, methylation, cna_data, clinical_data
)
pam50_labels, survival_times, survival_events, valid_survival, label_encoder = prepare_labels_and_survival(clinical_aligned)

pickle.dump({
    'gene_expr_scaled': gene_expr_scaled,
    'methylation_scaled': methylation_scaled,
    'cna_scaled': cna_scaled,
    'clinical_aligned': clinical_aligned,
    'pam50_labels': pam50_labels,
    'survival_times': survival_times,
    'survival_events': survival_events,
    'valid_survival': valid_survival,
    'label_encoder': label_encoder
}, open('preprocessed_data.pkl', 'wb'))

print("Data preprocessing completed!")
