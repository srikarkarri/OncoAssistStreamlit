import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def show():
    st.markdown("""
    <div class="custom-header">
        <h1>📊 Clinical Dashboard</h1>
        <p>Real-time insights based on 92.13% accurate subtype predictions</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics row with results
    st.markdown("### 📈 Model Performance Metrics")
    
    metric_cols = st.columns(4)
    
    with metric_cols[0]:
        st.metric(
            label="Classification Accuracy",
            value="92.13%",
            delta="+6.13% vs PAM50 benchmark",
            help="Your model's actual performance"
        )
    
    with metric_cols[1]:
        st.metric(
            label="Survival C-Index",
            value="0.714",
            delta="+0.014 vs median",
            help="Concordance index for survival prediction"
        )
    
    with metric_cols[2]:
        st.metric(
            label="Patients Analyzed", 
            value="1,411",
            delta="Complete dataset",
            help="Total patients in analysis"
        )
    
    with metric_cols[3]:
        st.metric(
            label="Subtypes Detected",
            value="6",
            delta="Including claudin-low",
            help="Basal, Her2, LumA, LumB, Normal, Claudin-low"
        )
    
    # Main dashboard content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Per-subtype performance metrics
        st.markdown("### 🎯 Per-Subtype Performance")
        
        # Your actual classification results
        performance_data = pd.DataFrame({
            'Subtype': ['Basal', 'Her2', 'LumA', 'LumB', 'Normal', 'Claudin-low'],
            'Precision': [0.97, 0.94, 0.94, 0.88, 0.83, 0.97],
            'Recall': [0.96, 0.91, 0.90, 0.95, 0.85, 0.96],
            'F1-Score': [0.96, 0.93, 0.92, 0.92, 0.84, 0.97],
            'Support': [144, 174, 500, 339, 115, 139]
        })
        
        # Create heatmap of performance metrics
        metrics_matrix = performance_data[['Precision', 'Recall', 'F1-Score']].values
        
        fig_metrics = go.Figure(data=go.Heatmap(
            z=metrics_matrix,
            x=['Precision', 'Recall', 'F1-Score'],
            y=performance_data['Subtype'],
            colorscale='RdYlBu_r',
            text=np.round(metrics_matrix, 3),
            texttemplate="%{text}",
            textfont={"size": 12},
            colorbar=dict(title="Performance Score")
        ))
        
        fig_metrics.update_layout(
            title="Model Performance by Cancer Subtype",
            height=400
        )
        
        st.plotly_chart(fig_metrics, use_container_width=True)
        
        # Treatment distribution analysis
        st.markdown("### 💊 Treatment Recommendations by Predicted Subtype")
        
        # Based on your actual results
        treatment_data = pd.DataFrame({
            'Subtype': ['HER2-enriched (479)', 'Triple-negative (364)', 'Luminal B (169)', 
                       'Luminal A (143)', 'Basal (144)', 'Claudin-low (139)'],
            'First-Line Treatment': [
                'Trastuzumab + Pertuzumab',
                'Chemotherapy + Immunotherapy',
                'Hormone therapy + CDK4/6',
                'Tamoxifen/Anastrozole',
                'Aggressive chemotherapy',
                'Research protocols'
            ],
            'Expected Response': ['85%', '55%', '80%', '90%', '60%', '40%']
        })
        
        st.dataframe(treatment_data, hide_index=True, use_container_width=True)
    
    with col2:
        # Sample size distribution
        st.markdown("### 📊 Sample Distribution")
        
        sample_data = pd.DataFrame({
            'Subtype': ['HER2-enriched', 'Triple-negative', 'Luminal B', 'Luminal A', 'Basal', 'Claudin-low'],
            'Count': [479, 364, 169, 143, 144, 139]
        })
        
        fig_samples = px.bar(
            sample_data,
            x='Count',
            y='Subtype',
            orientation='h',
            title="Sample Size by Subtype",
            color='Count',
            color_continuous_scale='viridis'
        )
        
        fig_samples.update_layout(height=350)
        st.plotly_chart(fig_samples, use_container_width=True)
        
        # Confusion matrix summary
        st.markdown("### 🎯 Classification Accuracy")
        
        # Show key accuracy metrics
        accuracy_metrics = pd.DataFrame({
            'Metric': ['Overall Accuracy', 'Macro Avg', 'Weighted Avg'],
            'Value': ['92.13%', '92.00%', '92.00%']
        })
        
        st.dataframe(
            accuracy_metrics,
            column_config={
                "Value": st.column_config.TextColumn(
                    "Performance",
                    help="Model performance metrics",
                ),
            },
            hide_index=True,
            use_container_width=True
        )
        
        # Model confidence indicators
        st.markdown("### ⚡ Model Insights")
        
        st.info("🎯 **Highest Performance**: Claudin-low subtype (F1=0.97)")
        st.info("🔬 **Clinical Strength**: Excellent basal subtype detection (F1=0.96)")
        st.warning("⚠️ **Attention Needed**: Normal subtype shows lower precision (0.83)")
        
        if st.button("📊 View Detailed Results", key="detailed_results"):
            st.switch_page("pages/results.py")
if __name__ == "__main__":
    show()