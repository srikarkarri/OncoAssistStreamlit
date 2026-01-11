import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class DrugSensitivityPredictor:
    """Predict drug sensitivities based on cancer subtypes"""
    
    def __init__(self):
        # Drug sensitivity database (based on clinical evidence)
        self.drug_sensitivity_db = {
            'Luminal A': {
                'Tamoxifen': 0.85, 'Anastrozole': 0.82, 'Letrozole': 0.80,
                'Palbociclib': 0.75, 'Chemotherapy': 0.45, 'Fulvestrant': 0.70,
                'Ribociclib': 0.72, 'Abemaciclib': 0.68
            },
            'Luminal B': {
                'Tamoxifen': 0.70, 'Chemotherapy': 0.75, 'Palbociclib': 0.80,
                'Fulvestrant': 0.65, 'Ribociclib': 0.72, 'Anastrozole': 0.68,
                'Letrozole': 0.66, 'Abemaciclib': 0.74
            },
            'HER2-enriched': {
                'Trastuzumab': 0.90, 'Pertuzumab': 0.85, 'T-DM1': 0.78,
                'Chemotherapy': 0.70, 'Tucatinib': 0.82, 'Neratinib': 0.65,
                'Lapatinib': 0.62, 'Margetuximab': 0.68
            },
            'Triple-negative': {
                'Chemotherapy': 0.60, 'Pembrolizumab': 0.55, 'Carboplatin': 0.58,
                'PARP_inhibitors': 0.40, 'Atezolizumab': 0.45, 'Sacituzumab': 0.52,
                'Capecitabine': 0.48, 'Bevacizumab': 0.42
            }
        }
    
    def predict_drug_sensitivity(self, subtypes):
        """Predict drug sensitivities for given subtypes - FIXED VERSION"""
        subtype_names = ['Luminal A', 'Luminal B', 'HER2-enriched', 'Triple-negative']
        
        # Get all unique drugs across all subtypes
        all_drugs = set()
        for drugs in self.drug_sensitivity_db.values():
            all_drugs.update(drugs.keys())
        all_drugs = sorted(all_drugs)
        
        # Create sensitivity matrix for each sample
        sensitivity_matrix = []
        for subtype_idx in subtypes:
            if subtype_idx < len(subtype_names):
                subtype_name = subtype_names[subtype_idx]
                sensitivities = []
                for drug in all_drugs:
                    sensitivity = self.drug_sensitivity_db[subtype_name].get(drug, 0.0)
                    sensitivities.append(sensitivity)
                sensitivity_matrix.append(sensitivities)
            else:
                # Handle unknown subtypes
                sensitivity_matrix.append([0.0] * len(all_drugs))
        
        return np.array(sensitivity_matrix), all_drugs
    
    def plot_drug_sensitivity_heatmap(self, subtypes, save_path='drug_sensitivity_heatmap.png'):
        """Create drug sensitivity heatmap - FIXED VERSION"""
        
        subtype_names = ['Luminal A', 'Luminal B', 'HER2-enriched', 'Triple-negative']
        
        # Get unique subtypes that exist in the data
        unique_subtypes = np.unique(subtypes)
        valid_subtypes = [s for s in unique_subtypes if s < len(subtype_names)]
        
        if len(valid_subtypes) == 0:
            print("No valid subtypes found for drug sensitivity analysis")
            return
        
        # Get all drugs
        all_drugs = set()
        for drugs in self.drug_sensitivity_db.values():
            all_drugs.update(drugs.keys())
        all_drugs = sorted(all_drugs)
        
        # Create average sensitivity matrix
        avg_sensitivity = []
        subtype_labels = []
        
        for subtype in valid_subtypes:
            subtype_name = subtype_names[subtype]
            sensitivities = []
            
            for drug in all_drugs:
                sensitivity = self.drug_sensitivity_db[subtype_name].get(drug, 0.0)
                sensitivities.append(sensitivity)
            
            avg_sensitivity.append(sensitivities)
            subtype_labels.append(subtype_name)
        
        # Convert to numpy array
        avg_sensitivity = np.array(avg_sensitivity)
        
        # Create the heatmap
        plt.figure(figsize=(14, 8))
        sns.heatmap(avg_sensitivity, 
                   xticklabels=all_drugs,
                   yticklabels=subtype_labels,
                   annot=True, fmt='.2f', cmap='RdYlBu_r',
                   cbar_kws={'label': 'Predicted Sensitivity Score'})
        plt.title('Drug Sensitivity Predictions by Cancer Subtype\n(Higher scores indicate better predicted response)', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Therapeutic Agents', fontsize=12)
        plt.ylabel('Cancer Subtypes', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Drug sensitivity heatmap saved to '{save_path}'")
        print(f"✓ Analyzed {len(subtype_labels)} subtypes with {len(all_drugs)} therapeutic agents")
        
        # Print summary statistics
        print("\nDrug Sensitivity Summary:")
        for i, subtype in enumerate(subtype_labels):
            avg_sens = np.mean(avg_sensitivity[i])
            print(f"  {subtype}: Average sensitivity = {avg_sens:.3f}")
