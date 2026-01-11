import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sys
import torch
import sqlite3
import json
from datetime import datetime
import base64
import os
from io import BytesIO
from reportlab.lib.pagesizes import letter, A4
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from MultiModalGATEnhancedold1 import MultiModalGAT, TreatmentRecommendationSystem

# Fix pickle compatibility for MultiModalGAT
sys.modules['__main__'].MultiModalGAT = MultiModalGAT

st.set_page_config(page_title="CancerAI Precision Oncology", layout="wide")

# Initialize database
def init_database():
    conn = sqlite3.connect('patient_records.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_name TEXT NOT NULL,
            patient_id TEXT UNIQUE NOT NULL,
            age INTEGER,
            gender TEXT,
            diagnosis_date TEXT,
            clinical_data TEXT,
            mutations_data TEXT,
            cna_data TEXT,
            expression_data TEXT,
            methylation_data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS analysis_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL,
            subtype TEXT,
            confidence REAL,
            survival_months INTEGER,
            survival_5y REAL,
            treatment_plan TEXT,
            key_genes TEXT,
            analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
        )
    ''')
    conn.commit()
    conn.close()

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'selected_patient' not in st.session_state:
    st.session_state.selected_patient = None

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

def save_patient_record(patient_name, patient_id, age, gender, clinical_file, mutations_file, cna_file, expression_file, methylation_file):
    """Save patient record to database"""
    conn = sqlite3.connect('patient_records.db')
    c = conn.cursor()
    
    try:
        # Convert dataframes to JSON strings for storage
        clinical_data = pd.read_csv(clinical_file).to_json() if clinical_file else None
        mutations_data = pd.read_csv(mutations_file).to_json() if mutations_file else None
        cna_data = pd.read_csv(cna_file).to_json() if cna_file else None
        expression_data = pd.read_csv(expression_file).to_json() if expression_file else None
        methylation_data = pd.read_csv(methylation_file).to_json() if methylation_file else None
        
        c.execute('''
            INSERT INTO patients (patient_name, patient_id, age, gender, diagnosis_date, clinical_data, mutations_data, cna_data, expression_data, methylation_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (patient_name, patient_id, age, gender, datetime.now().strftime('%Y-%m-%d'), clinical_data, mutations_data, cna_data, expression_data, methylation_data))
        
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def get_all_patients():
    """Get all patient records"""
    conn = sqlite3.connect('patient_records.db')
    df = pd.read_sql_query('SELECT id, patient_name, patient_id, age, gender, diagnosis_date, created_at FROM patients ORDER BY created_at DESC', conn)
    conn.close()
    return df

def get_patient_data(patient_id):
    """Get specific patient data"""
    conn = sqlite3.connect('patient_records.db')
    c = conn.cursor()
    c.execute('SELECT * FROM patients WHERE patient_id = ?', (patient_id,))
    result = c.fetchone()
    conn.close()
    
    if result:
        columns = ['id', 'patient_name', 'patient_id', 'age', 'gender', 'diagnosis_date', 
                  'clinical_data', 'mutations_data', 'cna_data', 'expression_data', 'methylation_data', 'created_at']
        return dict(zip(columns, result))
    return None

def save_analysis_results(patient_id, subtype, confidence, survival_months, survival_5y, treatment_plan, key_genes):
    """Save analysis results to database"""
    conn = sqlite3.connect('patient_records.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO analysis_results (patient_id, subtype, confidence, survival_months, survival_5y, treatment_plan, key_genes)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (patient_id, subtype, confidence, survival_months, survival_5y, json.dumps(treatment_plan), json.dumps(key_genes)))
    conn.commit()
    conn.close()

def generate_pdf_report(patient_data, analysis_results):
    """Generate PDF report of analysis results"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=1,  # Center alignment
        textColor=colors.HexColor('#7950f2')
    )
    story.append(Paragraph("CancerAI Analysis Report", title_style))
    story.append(Spacer(1, 20))
    
    # Patient Information
    story.append(Paragraph("Patient Information", styles['Heading2']))
    patient_info = [
        ['Patient Name:', patient_data['patient_name']],
        ['Patient ID:', patient_data['patient_id']],
        ['Age:', str(patient_data['age'])],
        ['Gender:', patient_data['gender']],
        ['Analysis Date:', datetime.now().strftime('%Y-%m-%d %H:%M')]
    ]
    
    patient_table = Table(patient_info, colWidths=[2*72, 4*72])
    patient_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(patient_table)
    story.append(Spacer(1, 20))
    
    # Diagnosis
    story.append(Paragraph("Diagnosis", styles['Heading2']))
    diagnosis_text = f"<b>PAM50 Subtype:</b> {analysis_results['subtype']}"
    story.append(Paragraph(diagnosis_text, styles['Normal']))
    story.append(Spacer(1, 15))
    
    # Prognosis
    story.append(Paragraph("Prognosis", styles['Heading2']))
    prognosis_text = f"<b>Median Survival:</b> {analysis_results['survival_months']} months<br/>"
    prognosis_text += f"<b>5-year Survival Probability:</b> {analysis_results['survival_5y']:.1f}%"
    story.append(Paragraph(prognosis_text, styles['Normal']))
    story.append(Spacer(1, 15))
    
    # Treatment Plan
    story.append(Paragraph("Treatment Plan", styles['Heading2']))
    treatment = analysis_results['treatment_plan']
    treatment_text = f"<b>First-line:</b> {', '.join(treatment.get('first_line', []))}<br/>"
    treatment_text += f"<b>Targeted Therapy:</b> {', '.join(treatment.get('targeted', []))}<br/>"
    treatment_text += f"<b>Prognosis Notes:</b> {treatment.get('prognosis', '')}<br/>"
    treatment_text += f"<b>Clinical Considerations:</b> {treatment.get('considerations', '')}"
    story.append(Paragraph(treatment_text, styles['Normal']))
    story.append(Spacer(1, 15))
    
    # Key Genes
    story.append(Paragraph("Key Genes", styles['Heading2']))
    genes_text = ', '.join(analysis_results['key_genes'])
    story.append(Paragraph(f"<b>Relevant Genes:</b> {genes_text}", styles['Normal']))
    
    # Footer
    story.append(Spacer(1, 50))
    footer_text = "© 2025 CancerAI Lab - Precision Oncology Dashboard"
    story.append(Paragraph(footer_text, styles['Normal']))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

# Initialize database
init_database()

# Main App
st.markdown(
    """
    <h1 style='font-size:2.5rem; font-weight:700; color:#7950f2; margin-bottom:0.2em;'>CancerAI: Precision Oncology Dashboard</h1>
    <div style='font-size:1.2rem; color:#444; margin-bottom:2em;'>Upload patient data, manage patient records, and generate comprehensive analysis reports.</div>
    """, unsafe_allow_html=True
)

# Create tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["📁 Upload New Patient", "👥 Patient Records", "🔬 Analysis Results"])

with tab1:
    st.header("Upload New Patient Data")
    
    # Patient information form
    with st.form("patient_info"):
        col1, col2 = st.columns(2)
        with col1:
            patient_name = st.text_input("Patient Name", placeholder="Enter patient name")
            patient_id = st.text_input("Patient ID", placeholder="Enter unique patient ID")
        with col2:
            age = st.number_input("Age", min_value=1, max_value=120, value=50)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        
        # File uploads
        st.subheader("Upload Patient Data Files (all required)")
        col1, col2, col3 = st.columns(3)
        with col1:
            clinical_file = st.file_uploader("Clinical CSV", type="csv", key="clinical")
            mutations_file = st.file_uploader("Mutations CSV", type="csv", key="mutations")
        with col2:
            cna_file = st.file_uploader("CNA CSV", type="csv", key="cna")
            expression_file = st.file_uploader("Expression CSV", type="csv", key="expression")
        with col3:
            methylation_file = st.file_uploader("Methylation CSV", type="csv", key="methylation")
        
        submitted = st.form_submit_button("Save Patient Record")
        
        if submitted:
            if all([patient_name, patient_id, clinical_file, mutations_file, cna_file, expression_file, methylation_file]):
                success = save_patient_record(patient_name, patient_id, age, gender, clinical_file, mutations_file, cna_file, expression_file, methylation_file)
                if success:
                    st.success(f"Patient record for {patient_name} (ID: {patient_id}) saved successfully!")
                else:
                    st.error("Patient ID already exists. Please use a unique Patient ID.")
            else:
                st.error("Please fill in all required fields and upload all files.")

with tab2:
    st.header("Patient Records")
    
    patients_df = get_all_patients()
    
    if not patients_df.empty:
        # Display patients in a nice format
        for _, patient in patients_df.iterrows():
            with st.expander(f"👤 {patient['patient_name']} (ID: {patient['patient_id']}) - {patient['diagnosis_date']}"):
                col1, col2, col3 = st.columns([2, 2, 1])
                with col1:
                    st.write(f"**Name:** {patient['patient_name']}")
                    st.write(f"**Age:** {patient['age']}")
                with col2:
                    st.write(f"**Gender:** {patient['gender']}")
                    st.write(f"**Created:** {patient['created_at'][:10]}")
                with col3:
                    if st.button(f"Select & Analyze", key=f"select_{patient['patient_id']}"):
                        st.session_state.selected_patient = patient['patient_id']
                        st.rerun()
    else:
        st.info("No patient records found. Please upload patient data in the 'Upload New Patient' tab.")

with tab3:
    st.header("Analysis Results")
    
    if st.session_state.selected_patient:
        patient_data = get_patient_data(st.session_state.selected_patient)
        
        if patient_data:
            # Display selected patient info
            st.info(f"Selected Patient: **{patient_data['patient_name']}** (ID: {patient_data['patient_id']})")
            
            if st.button("🔬 Run Analysis", type="primary"):
                with st.spinner("Running analysis..."):
                    # Reconstruct dataframes from stored JSON
                    clinical = pd.read_json(patient_data['clinical_data'])
                    mutations = pd.read_json(patient_data['mutations_data'])
                    cna = pd.read_json(patient_data['cna_data'])
                    expression = pd.read_json(patient_data['expression_data'])
                    methylation = pd.read_json(patient_data['methylation_data'])

                    # Run analysis
                    X = preprocess_patient_data(clinical, expression, cna, methylation, mutations)
                    model = load_model()
                    sub_idx = int(model.predict(X)[0])
                    proba = model.predict_proba(X)[0]
                    conf = float(np.max(proba))

                    # Labels & look-ups
                    subtypes = ['Luminal A', 'Luminal B', 'HER2-enriched', 'Triple-negative']
                    subtype = subtypes[sub_idx]
                    fiveyr = [95.6, 91.8, 86.5, 78.4][sub_idx]
                    median_m = [140, 115, 90, 70][sub_idx]

                    treat_sys = TreatmentRecommendationSystem()
                    plan = treat_sys.generate_treatment_recommendations([sub_idx])[0]

                    def key_genes(sub_idx):
                        lookup = {
                            0: ["ESR1", "PGR", "CCND1", "CDK4", "CDK6"],
                            1: ["ESR1", "PGR", "CCND1", "CDK4", "CDK6", "MKI67"],
                            2: ["ERBB2", "ESR1", "PGR", "EGFR"],
                            3: ["BRCA1", "BRCA2", "TP53", "PTEN", "PIK3CA"]
                        }
                        return lookup.get(sub_idx, [])

                    genes = key_genes(sub_idx)

                    # Store results in session state
                    st.session_state.analysis_results = {
                        'patient_data': patient_data,
                        'subtype': subtype,
                        'confidence': conf,
                        'survival_months': median_m,
                        'survival_5y': fiveyr,
                        'treatment_plan': plan,
                        'key_genes': genes,
                        'sub_idx': sub_idx
                    }

                    # Save to database
                    save_analysis_results(patient_data['patient_id'], subtype, conf, median_m, fiveyr, plan, genes)

            # Display results
            if st.session_state.analysis_results:
                results = st.session_state.analysis_results
                
                # Diagnosis card (without confidence score)
                diag_html = f"""<h2 style='margin-top:0;color:#7950f2;'>Diagnosis</h2>
<p style='font-size:1.3rem;'>
  <b>PAM50 Subtype:</b>
  <span style='color:#2ca02c;font-weight:600;'>{results['subtype']}</span>
</p>"""
                floating_card(diag_html, bgcolor="#ffffff")

                # Prognosis card
                prog_html = f"""<h2 style='margin-top:0;color:#d6336c;'>Prognosis</h2>
<p style='font-size:1.2rem;margin-bottom:0.6rem;'>
  <b>Median survival:</b> {results['survival_months']}&nbsp;months
</p>
<p style='font-size:1.2rem;'><b>5-year survival probability:</b> {results['survival_5y']:.1f}%</p>"""
                floating_card(prog_html, bgcolor="#ffffff")
                plot_kaplan_meier_curve(results['subtype'])

                # Treatment plan card
                plan = results['treatment_plan']
                treat_html = f"""<h2 style='margin-top:0;color:#228be6;'>Treatment Plan</h2>
<b>First-line:</b> {', '.join(plan.get('first_line', []))}<br>
<b>Targeted&nbsp;therapy:</b> {', '.join(plan.get('targeted', []))}<br>
<b>Prognosis notes:</b> {plan.get('prognosis', '')}<br>
<b>Clinical considerations:</b> {plan.get('considerations', '')}"""
                floating_card(treat_html, bgcolor="#f8f9fa")

                # Key genes card
                gene_badges = " ".join(
                    [f"<span style='background:#e7e9ff;border-radius:0.4rem;padding:0.20rem 0.55rem;margin:0.2rem;display:inline-block;font-weight:600;color:#4c4f69;'>{g}</span>"
                     for g in results['key_genes']]
                )
                gene_html = f"""<h2 style='margin-top:0;color:#845ef7;'>Key Genes</h2>
<div style='font-size:1.1rem;'>{gene_badges}</div>"""
                floating_card(gene_html, bgcolor="#ffffff")

                # PDF Download Button
                st.markdown("---")
                col1, col2, col3 = st.columns([1, 1, 1])
                with col2:
                    if st.button("📥 Download PDF Report", type="secondary", use_container_width=True):
                        pdf_buffer = generate_pdf_report(results['patient_data'], results)
                        st.download_button(
                            label="📄 Download Analysis Report",
                            data=pdf_buffer,
                            file_name=f"CancerAI_Report_{results['patient_data']['patient_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )

                st.success("Analysis complete! You can now download the PDF report above.")
    else:
        st.info("Please select a patient from the 'Patient Records' tab to run analysis.")

# Footer
st.markdown("---")
st.markdown("<div style='font-size:0.9em; color:#888; text-align:right;'>© 2025 CancerAI Lab</div>", unsafe_allow_html=True)
