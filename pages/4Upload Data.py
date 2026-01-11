import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sys
import torch
from MultiModalGATEnhancedold1 import MultiModalGAT, TreatmentRecommendationSystem

# Fix pickle compatibility for MultiModalGAT
sys.modules['__main__'].MultiModalGAT = MultiModalGAT

st.set_page_config(page_title="CancerAI Precision Oncology", layout="wide")

def floating_card(content, bgcolor="#fff", border="#e0e0e0"):
    st.markdown(
        f"""
        <div style="
            background: rgba(255,255,255,0.85);
            border-radius: 1.5rem;
            box-shadow: 0 8px 32px 0 rgba(31,38,135,0.15);
            border: 1px solid {border};
            padding: 2rem 2rem 1.5rem 2rem;
            margin-bottom: 2rem;
            backdrop-filter: blur(8px);
        ">
        {content}
        </div>
        """, unsafe_allow_html=True
    )

@st.cache_resource
def load_model():
    with open("trained_model_fixed.pkl", "rb") as f:
        obj = pickle.load(f)
        model = obj['model']
        model.load_state_dict(obj['model_state_dict'])
        model.eval()
    return model

def preprocess_patient_data(clinical, expression, cna, methylation, mutations):
    features = []
    clin_num = clinical.select_dtypes(include=[np.number]).values.flatten()
    features.append(clin_num)
    expr_num = expression.iloc[:,2:].values.flatten()
    features.append(expr_num)
    cna_num = cna.iloc[:,2:].values.flatten()
    features.append(cna_num)
    meth_num = methylation.iloc[:,1:].values.flatten()
    features.append(meth_num)
    mut_count = len(mutations) if mutations is not None else 0
    features.append(np.array([mut_count]))
    X = np.concatenate(features).astype(np.float32)
    return X.reshape(1,-1)

def plot_kaplan_meier_curve(subtype):
    import matplotlib.pyplot as plt
    from lifelines import KaplanMeierFitter

    # Subtype-specific survival curves (priors)
    subtype_curves = {
        "Luminal A": (0.956, 0.90),
        "Luminal B": (0.918, 0.80),
        "HER2-enriched": (0.865, 0.70),
        "Triple-negative": (0.784, 0.60),
    }
    months = np.arange(0, 121, 6)
    surv_5y, surv_10y = subtype_curves.get(subtype, (0.8, 0.6))
    y = np.linspace(surv_5y, surv_10y, len(months))
    y = np.concatenate([[1.0], y])
    months = np.concatenate([[0], months])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.step(months, y, where="post", color="#2ca02c", lw=3)
    ax.fill_between(months, y - 0.05, y + 0.05, color="#2ca02c", alpha=0.15)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, 120)
    ax.set_xlabel("Time (months)")
    ax.set_ylabel("Survival Probability")
    ax.set_title(f"Kaplan-Meier Survival Curve: {subtype}")
    st.pyplot(fig, use_container_width=True)

def plot_drug_sensitivity_heatmap():
    st.info("Drug sensitivity heatmap functionality to be implemented here.")
    st.image("drug_sensitivity_heatmap.jpg", caption="Drug Sensitivity Heatmap", use_container_width=True)

def get_targeted_genes(subtype, omics_data):
    """Get list of targeted genes based on subtype and available omics data"""
    target_genes = {
        0: ["ESR1", "PGR", "CCND1", "CDK4", "CDK6"],  # Luminal A
        1: ["ESR1", "PGR", "CCND1", "CDK4", "CDK6", "MKI67"],  # Luminal B
        2: ["ERBB2", "ESR1", "PGR", "EGFR"],  # HER2-enriched
        3: ["BRCA1", "BRCA2", "TP53", "PTEN", "PIK3CA"]  # Triple-negative
    }
    
    genes = target_genes.get(subtype, [])
    return genes

st.markdown(
    """
    <h1 style='font-size:2.5rem; font-weight:700; color:#7950f2; margin-bottom:0.2em;'>CancerAI: Precision Oncology Dashboard</h1>
    <div style='font-size:1.2rem; color:#444; margin-bottom:2em;'>Upload a patient's multi-omics data to predict cancer subtype, survival, and treatment plan.</div>
    """, unsafe_allow_html=True
)

with st.expander("Upload Patient Data Files (all required)", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        clinical_file = st.file_uploader("Clinical CSV", type="csv", key="clinical")
        mutations_file = st.file_uploader("Mutations CSV", type="csv", key="mutations")
    with col2:
        cna_file = st.file_uploader("CNA CSV", type="csv", key="cna")
        gene_panel_file = st.file_uploader("Gene Panel CSV (optional)", type="csv", key="genepanel")
    with col3:
        expression_file = st.file_uploader("Expression CSV", type="csv", key="expression")
        methylation_file = st.file_uploader("Methylation CSV", type="csv", key="methylation")

    ready = all([clinical_file, mutations_file, cna_file, expression_file, methylation_file])

if st.button("Run Analysis", disabled=not ready):
    # -------------------- 1. Model inference -------------------------------
    clinical    = pd.read_csv(clinical_file)
    mutations   = pd.read_csv(mutations_file)
    cna         = pd.read_csv(cna_file)
    expression  = pd.read_csv(expression_file)
    methylation = pd.read_csv(methylation_file)

    X          = preprocess_patient_data(clinical, expression, cna, methylation, mutations)
    model      = load_model()
    sub_idx    = int(model.predict(X)[0])
    proba      = model.predict_proba(X)[0]
    conf       = float(np.max(proba))

    # Labels & look-ups
    subtypes = ['Luminal A', 'Luminal B', 'HER2-enriched', 'Triple-negative']
    subtype  = subtypes[sub_idx]
    fiveyr   = [95.6, 91.8, 86.5, 78.4][sub_idx]  # %
    median_m = [140, 115,  90,  70][sub_idx]      # rough median-survival months

    treat_sys = TreatmentRecommendationSystem()
    plan      = treat_sys.generate_treatment_recommendations([sub_idx])[0]

    # -------------------- 2. Diagnosis card -------------------------------
    diag_html = f"""<h2 style='margin-top:0;color:#7950f2;'>Diagnosis</h2>
<p style='font-size:1.3rem;'>
  <b>PAM50 Subtype:</b>
  <span style='color:#2ca02c;font-weight:600;'>{subtype}</span>
</p>
<p style='font-size:1.2rem;'><b>Model confidence:</b> {conf:.1%}</p>"""
    floating_card(diag_html, bgcolor="#ffffff")

    # -------------------- 3. Prognosis card -------------------------------
    prog_html = f"""<h2 style='margin-top:0;color:#d6336c;'>Prognosis</h2>
<p style='font-size:1.2rem;margin-bottom:0.6rem;'>
  <b>Median survival:</b> {median_m}&nbsp;months
</p>
<p style='font-size:1.2rem;'><b>5-year survival probability:</b> {fiveyr:.1f}%</p>"""
    floating_card(prog_html, bgcolor="#ffffff")
    plot_kaplan_meier_curve(subtype)   # graph appears directly below the Prognosis card

    # -------------------- 4. Treatment-plan card --------------------------
    treat_html = f"""<h2 style='margin-top:0;color:#228be6;'>Treatment plan</h2>
<b>First-line:</b>  {', '.join(plan.get('first_line', []))}<br>
<b>Targeted&nbsp;therapy:</b>  {', '.join(plan.get('targeted', []))}<br>
<b>Prognosis notes:</b> {plan.get('prognosis', '')}<br>
<b>Clinical considerations:</b> {plan.get('considerations', '')}"""
    floating_card(treat_html, bgcolor="#f8f9fa")

    # -------------------- 5. Key-genes card -------------------------------
    def key_genes(sub_idx):
        lookup = {
            0:["ESR1","PGR","CCND1","CDK4","CDK6"],
            1:["ESR1","PGR","CCND1","CDK4","CDK6","MKI67"],
            2:["ERBB2","ESR1","PGR","EGFR"],
            3:["BRCA1","BRCA2","TP53","PTEN","PIK3CA"]
        }
        return lookup.get(sub_idx, [])
    genes = key_genes(sub_idx)

    gene_badges = " ".join(
        [f"<span style='background:#e7e9ff;border-radius:0.4rem;padding:0.20rem 0.55rem;margin:0.2rem;display:inline-block;font-weight:600;color:#4c4f69;'>{g}</span>"
         for g in genes]
    )

    gene_html = f"""<h2 style='margin-top:0;color:#845ef7;'>Key genes</h2>
<div style='font-size:1.1rem;'>{gene_badges}</div>"""
    floating_card(gene_html, bgcolor="#ffffff")

    st.success("Analysis complete.")

else:
    st.info("Upload all required files and click 'Run Analysis' to begin.")

st.markdown("<div style='font-size:0.9em; color:#888; text-align:right;'>© 2025 CancerAI Lab</div>", unsafe_allow_html=True)
