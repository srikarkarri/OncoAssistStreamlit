import streamlit as st
import os
import json
from openai import OpenAI
import warnings
warnings.filterwarnings('ignore')

# Import database functionality
try:
    from database import db_manager
    DATABASE_AVAILABLE = db_manager.is_connected()
except Exception as e:
    DATABASE_AVAILABLE = False

# Configure page
st.set_page_config(
    page_title="AI Assistant - Onco Assist",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .user-message {
        background: #2E86AB;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        margin-left: 20%;
    }
    .assistant-message {
        background: white;
        color: #333;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        margin-right: 20%;
        border-left: 4px solid #2E86AB;
    }
    .guideline-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-left: 4px solid #A23B72;
    }
    .quick-question {
        background: #e3f2fd;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.25rem;
        cursor: pointer;
        border: 1px solid #2196F3;
        display: inline-block;
    }
    .quick-question:hover {
        background: #bbdefb;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'openai_client' not in st.session_state:
    st.session_state.openai_client = None

# Initialize OpenAI client

def initialize_openai():
    """Initialize OpenAI client with priority: st.secrets -> env var -> session input"""
    try:
        OPENAI_API_KEY = None
        # 1) Streamlit Cloud / secrets.toml
        if "OPENAI_API_KEY" in st.secrets:
            OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
        # 2) Local env (export/setx OPENAI_API_KEY=...)
        if not OPENAI_API_KEY:
            OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
        # 3) Optional: user-provided (stored only in session)
        if not OPENAI_API_KEY:
            OPENAI_API_KEY = st.session_state.get("OPENAI_API_KEY_INPUT")
        if OPENAI_API_KEY:
            st.session_state.openai_client = OpenAI(api_key=OPENAI_API_KEY)
            return True
        return False
    except Exception as e:
        st.error(f"Error initializing OpenAI client: {str(e)}")
        return False

def get_ai_response(question, context=""):
    """Get AI response following medical guidelines"""
    if not st.session_state.openai_client:
        return "OpenAI client not initialized. Please check your API key."
    
    try:
        system_prompt = """You are an AI assistant specializing in oncology and breast cancer treatment guidelines. 
        You follow NCCN (National Comprehensive Cancer Network), ESMO (European Society for Medical Oncology), 
        and other international oncology guidelines.

        Your responses should be:
        1. Evidence-based and cite relevant guidelines
        2. Clear and professionally written for healthcare providers
        3. Include appropriate caveats about clinical judgment
        4. Mention when consultation with specialists is recommended
        5. Focus on breast cancer subtypes, treatment options, and clinical decision-making

        Always emphasize that AI recommendations should supplement, not replace, clinical judgment.
        """
        
        response = st.session_state.openai_client.chat.completions.create(
            model="gpt-4o",  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{context}\n\nQuestion: {question}"}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        response_text = response.choices[0].message.content
        
        # Log interaction to database if available
        if DATABASE_AVAILABLE:
            try:
                db_manager.log_ai_interaction(
                    question=question,
                    response=response_text,
                    context=context
                )
            except Exception as log_error:
                pass  # Don't fail the response if logging fails
        
        return response_text
    
    except Exception as e:
        return f"Error generating response: {str(e)}"

def main():
    st.title("🤖 AI Clinical Assistant")
    st.markdown("Evidence-based oncology guidance following NCCN, ESMO, and international guidelines")
    
    # Initialize OpenAI if not already done
    if not st.session_state.openai_client:
        if not initialize_openai():
            st.error("⚠️ OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
            st.markdown("""
            <div class="guideline-card">
                <h4>🔧 Setup Required</h4>
                <p>To use the AI Assistant, you need to provide an OpenAI API key.</p>
                <p>The assistant will provide evidence-based oncology guidance following:</p>
                <ul>
                    <li>NCCN Guidelines</li>
                    <li>ESMO Guidelines</li>
                    <li>International oncology standards</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            return
    
    # Quick Questions
    st.markdown("### 🚀 Quick Questions")
    
    quick_questions = [
        "What are the treatment options for Triple-negative breast cancer?",
        "When should Oncotype DX testing be considered?",
        "What is the role of CDK4/6 inhibitors in Luminal A breast cancer?",
        "How do NCCN guidelines recommend staging workup for newly diagnosed breast cancer?",
        "What are the indications for neoadjuvant chemotherapy?",
        "When is genetic testing recommended for breast cancer patients?",
        "What are the fertility preservation options for young breast cancer patients?",
        "How should HER2-positive breast cancer be treated according to ESMO guidelines?"
    ]
    
    cols = st.columns(2)
    for i, question in enumerate(quick_questions):
        with cols[i % 2]:
            if st.button(question, key=f"quick_{i}"):
                st.session_state.chat_history.append({"role": "user", "content": question})
                with st.spinner("Getting AI response..."):
                    response = get_ai_response(question)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.rerun()
    
    st.markdown("---")
    
    # Chat Interface
    st.markdown("### 💬 Ask Your Question")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_question = st.text_input(
            "Enter your oncology question:",
            placeholder="e.g., What are the latest NCCN recommendations for HER2-positive breast cancer treatment?"
        )
    
    with col2:
        if st.button("Ask Assistant", type="primary"):
            if user_question:
                st.session_state.chat_history.append({"role": "user", "content": user_question})
                with st.spinner("Getting AI response..."):
                    # Add context from current patient if available
                    context = ""
                    if 'predictions' in st.session_state and st.session_state.predictions:
                        subtype = st.session_state.predictions.get('predicted_subtype', '')
                        context = f"Current patient context: Predicted subtype is {subtype}."
                    
                    response = get_ai_response(user_question, context)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.rerun()
    
    # Chat History
    if st.session_state.chat_history:
        st.markdown("### 📜 Conversation History")
        
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    st.markdown(f"""
                    <div class="user-message">
                        <strong>You:</strong> {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="assistant-message">
                        <strong>🤖 AI Assistant:</strong><br>
                        {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Guidelines Information
    st.markdown("---")
    st.markdown("### 📚 Clinical Guidelines Coverage")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="guideline-card">
            <h4>🏥 NCCN Guidelines</h4>
            <p><strong>National Comprehensive Cancer Network</strong></p>
            <ul>
                <li>Breast Cancer Treatment Guidelines</li>
                <li>Genetic/Familial High-Risk Assessment</li>
                <li>Survivorship Guidelines</li>
                <li>Supportive Care Guidelines</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="guideline-card">
            <h4>🔬 ESMO Guidelines</h4>
            <p><strong>European Society for Medical Oncology</strong></p>
            <ul>
                <li>Early Breast Cancer Guidelines</li>
                <li>Advanced Breast Cancer Guidelines</li>
                <li>Biomarker Testing Recommendations</li>
                <li>Cardio-Oncology Guidelines</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="guideline-card">
            <h4>🌍 International Guidelines</h4>
            <p><strong>Global Oncology Standards</strong></p>
            <ul>
                <li>WHO Classification Updates</li>
                <li>ASCO Clinical Practice Guidelines</li>
                <li>St. Gallen Consensus</li>
                <li>AJCC Staging Guidelines</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="guideline-card">
            <h4>🎯 Specialized Topics</h4>
            <p><strong>Focused Clinical Areas</strong></p>
            <ul>
                <li>Precision Medicine Approaches</li>
                <li>Immunotherapy Guidelines</li>
                <li>Fertility Preservation</li>
                <li>Geriatric Oncology</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Example Queries
    st.markdown("---")
    st.markdown("### 💡 Example Queries")
    
    example_queries = [
        {
            "category": "Treatment Selection",
            "examples": [
                "What factors determine chemotherapy selection for Luminal B breast cancer?",
                "When should anti-HER2 therapy be discontinued?",
                "What are the contraindications for CDK4/6 inhibitors?"
            ]
        },
        {
            "category": "Biomarker Testing",
            "examples": [
                "Which patients should receive multigene assays?",
                "How do I interpret Oncotype DX scores?",
                "What is the clinical utility of PIK3CA mutation testing?"
            ]
        },
        {
            "category": "Survivorship",
            "examples": [
                "What are the surveillance recommendations for breast cancer survivors?",
                "How should treatment-related toxicities be managed?",
                "When should genetic counseling be referred?"
            ]
        }
    ]
    
    for query_group in example_queries:
        st.markdown(f"**{query_group['category']}:**")
        for example in query_group['examples']:
            st.markdown(f"• {example}")
        st.markdown("")
    
    # Disclaimer
    st.markdown("---")
    st.markdown("""
    <div style="background: #fff3cd; padding: 1rem; border-radius: 5px; border-left: 4px solid #ffc107;">
        <h5>⚠️ Important Clinical Disclaimer</h5>
        <p>This AI assistant provides information based on published clinical guidelines and evidence-based medicine. 
        However, it should not replace professional medical judgment, clinical experience, or individualized patient assessment. 
        Always consider:</p>
        <ul>
            <li>Patient-specific factors and comorbidities</li>
            <li>Local institutional protocols and preferences</li>
            <li>Most recent guideline updates and emerging evidence</li>
            <li>Multidisciplinary team consultation when appropriate</li>
        </ul>
        <p><strong>For any clinical decision-making, consult with qualified healthcare professionals and refer to the most current official guidelines.</strong></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
