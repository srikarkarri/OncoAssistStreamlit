import pickle
import pandas as pd


# Load the graph data from the pickle file
pickle_file = 'graph_data.pkl'
with open(pickle_file, 'rb') as f:
    graph_data = pickle.load(f)

with open('graph_data.pkl', 'rb') as f:
    data_dict = pickle.load(f)

graph_data = data_dict['graph_data']  # This is your PyG Data object

# Convert node features to numpy array
features = graph_data.x.cpu().numpy()

# If you know the original feature names (e.g., gene names), use them here.
# Otherwise, use generic names:
feature_columns = [f'feature_{i}' for i in range(features.shape[1])]

# Build DataFrame
df = pd.DataFrame(features, columns=feature_columns)
df['pam50_label'] = graph_data.y.cpu().numpy()
df['survival_time'] = graph_data.survival_time.cpu().numpy()
df['event'] = graph_data.event.cpu().numpy()

# Save to Excel
df.to_excel('patients_table.xlsx', index=False)

print("Saved patient data to patients_table.xlsx")

import pickle

# Load preprocessed data to get original feature names
with open('preprocessed_data.pkl', 'rb') as f:
    preproc = pickle.load(f)

gene_names = list(preproc['gene_expr_scaled'].columns)
meth_names = list(preproc['methylation_scaled'].columns)
cna_names = list(preproc['cna_scaled'].columns)

# Concatenate in the same order as your combined_features
all_feature_names = gene_names + meth_names + cna_names

with open('graph_data.pkl', 'rb') as f:
    data_dict = pickle.load(f)

selected_indices = data_dict['selected_indices']
selected_feature_names = [all_feature_names[i] for i in selected_indices]

import pandas as pd

# Create DataFrame mapping index to real feature name
feature_df = pd.DataFrame({
    'feature_index': list(range(len(selected_feature_names))),
    'feature_name': selected_feature_names
})

feature_df.to_excel('feature_names_table.xlsx', index=False)
print("Saved real feature names to feature_names_table.xlsx")

features = graph_data.x.cpu().numpy()
df = pd.DataFrame(features, columns=selected_feature_names)
df['pam50_label'] = graph_data.y.cpu().numpy()
df['survival_time'] = graph_data.survival_time.cpu().numpy()
df['event'] = graph_data.event.cpu().numpy()

df.to_excel('patients_table.xlsx', index=False)
print("Saved patient data with real feature names to patients_table.xlsx")


# import pickle
# import pandas as pd

# # --- Load the saved graph data dictionary ---
# with open('graph_data.pkl', 'rb') as f:
#     data_dict = pickle.load(f)

# graph_data = data_dict['graph_data']  # PyG Data object
# selected_indices = data_dict['selected_indices']  # Indices of selected features

# # --- 1. Export patient-level data ---
# features = graph_data.x.cpu().numpy()
# num_features = features.shape[1]

# # If you have actual feature names, provide them here:
# # Example: all_feature_names = ['TP53', 'BRCA1', ...]  # length = total features before LASSO
# # feature_names = [all_feature_names[i] for i in selected_indices]
# # If not, use generic names:
# feature_names = [f'feature_{i}' for i in range(num_features)]

# # Build DataFrame for patients
# df = pd.DataFrame(features, columns=feature_names)
# df['pam50_label'] = graph_data.y.cpu().numpy()
# df['survival_time'] = graph_data.survival_time.cpu().numpy()
# df['event'] = graph_data.event.cpu().numpy()

# df.to_excel('patients_table.xlsx', index=False)
# print("Saved patient data to patients_table.xlsx")

# # --- 2. Export feature index-to-name mapping ---
# feature_df = pd.DataFrame({
#     'feature_index': list(range(num_features)),
#     'feature_name': feature_names
# })

# feature_df.to_excel('feature_names_table.xlsx', index=False)
# print("Saved feature names to feature_names_table.xlsx")