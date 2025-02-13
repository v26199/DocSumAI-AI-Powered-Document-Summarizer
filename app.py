import streamlit as st
import urllib.parse
import sys
import subprocess
import logging
import spacy

from model2 import (
    extract_text,
    process_uploaded_file,
    generate_llm_summary,
    generate_nlp_summary,
    generate_extractive_summary
)
from evaluation import evaluate_summary, display_evaluation

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Download SpaCy model if not available
try:
    nlp_extract = spacy.load('en_core_web_lg')
except OSError:
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_lg"])
    nlp_extract = spacy.load('en_core_web_lg')

def handle_file_upload():
    """Handle file upload"""
    uploaded_file = st.file_uploader("Choose file", type=["pdf", "docx", "txt"])
    return uploaded_file

def generate_summary(
    summarization_method: str, 
    text: str, 
    summary_length: int, 
    extract_sentences: int = None, 
    llm_model: str = None
) -> str:
    """Generate summary based on the selected method"""
    if summarization_method == "LLM Summarization":
        return generate_llm_summary(text, llm_model, summary_length)
    elif summarization_method == "NLP Pipeline Summarization":
        return generate_nlp_summary(text, summary_length)
    elif summarization_method == "Extractive Text Summarization":
        return generate_extractive_summary(text, extract_sentences)
    else:
        return "No summarization method selected."

def process_input(content) -> str:
    """
    Process the input content.
    If content is a string (e.g., loaded from URL), it is returned as is.
    Otherwise, extract text from the uploaded file.
    """
    if isinstance(content, str):
        return content
    else:
        return extract_text(content)

def main():
    st.title("📚 AI Document Summarizer")
    st.markdown("<h3 style='color: #4CAF50; font-weight: bold;'>LLMs, NLP Pipeline, and Extractive Summarization</h3>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Input Section: Allow user to choose file upload or URL
    input_method = st.radio("Input Source:", ["📁 Upload Document", "🌐 Enter URL"], horizontal=True)
    content = None
    if input_method == "📁 Upload Document":
        uploaded_file = handle_file_upload()
        if uploaded_file:
            content = process_uploaded_file(uploaded_file, uploaded_file.name)
    else:
        url = st.text_input("Enter URL:")
        is_valid_url = False
        if url:
            try:
                result = urllib.parse.urlparse(url)
                is_valid_url = all([result.scheme, result.netloc])
                if not is_valid_url:
                    st.error("❌ Invalid URL. Please enter a valid URL (e.g., https://example.com).")
            except ValueError:
                st.error("❌ Invalid URL. Please enter a valid URL.")
        
        if st.button("Load Data from Web"):
            if is_valid_url:
                with st.spinner("🔄 Loading data from the web..."):
                    try:
                        loaded_text = extract_text(url)
                        st.session_state.loaded_text = loaded_text
                        st.success("✅ Data loaded successfully!")
                    except Exception as e:
                        st.error(f"❌ Error loading web data: {e}")
            else:
                st.warning("⚠️ Please enter a valid URL before clicking the button.")
        
        if 'loaded_text' in st.session_state:
            st.text_area("Loaded Text Preview", value=st.session_state.loaded_text, height=300)
            content = st.session_state.loaded_text
    
    # Summarization Method Selection
    summarization_method = st.radio(
        "Select Summarization Method:",
        ["LLM Summarization", "NLP Pipeline Summarization", "Extractive Text Summarization"],
        horizontal=True
    )
    
    # Summary Settings
    with st.expander("⚙️ Summary Settings"):
        llm_model = None
        extract_sentences = None
        if summarization_method == "LLM Summarization":
            llm_model = st.selectbox("LLM Model (Groq API):", 
                                     ["llama3-70b-8192", "mixtral-8x7b-32768", "llama-3.1-8b-instant", "gemma2-9b-it"])
        summary_length = st.slider("Summary Length (words):", 100, 2000, 500)
        if summarization_method == "Extractive Text Summarization":
            extract_sentences = st.slider("Number of Extractive Sentences:", 1, 10, 3)
    
    # Process input and run summarization
    if content and st.button("Summarize Document"):
        with st.spinner("🔍 Analyzing content..."):
            try:
                text = process_input(content)
                
                if not text.strip():
                    st.warning("No text extracted from the document. Please check the file or URL.")
                    return
                
                summary = generate_summary(summarization_method, text, summary_length, extract_sentences, llm_model)
                
                st.markdown(f"### 📋 {summarization_method}")
                st.markdown(summary, unsafe_allow_html=True)
                
                scores = evaluate_summary(text, summary)
                st.subheader("📊 Evaluation Metrics")
                plot = display_evaluation(scores)
                if plot:
                    st.pyplot(plot, clear_figure=True)
                
                eval_text = f"""
                **Evaluation Findings:**  
                - **BLEU:** {scores.get('BLEU', 0):.4f}  
                  *Measures n-gram overlap between summary and original text*  
                - **ROUGE-1:** {scores.get('ROUGE-1', 0):.4f}  
                  *Assesses unigram recall*  
                - **ROUGE-2:** {scores.get('ROUGE-2', 0):.4f}  
                  *Assesses bigram recall*  
                - **ROUGE-L:** {scores.get('ROUGE-L', 0):.4f}  
                  *Measures longest common subsequence*  
                - **METEOR:** {scores.get('METEOR', 0):.4f}  
                  *Considers synonymy and alignment*  
                **Interpretation:**  
                Higher scores indicate better alignment with source content.
                """
                st.markdown("### 📝 Evaluation Summary")
                st.markdown(eval_text)
                
            except Exception as e:
                logging.error(f"Error processing content: {e}")
                st.error(f"Processing error: {str(e)}")
    
    st.markdown("---")
    st.write(
        """<center>
<strong>FEATURES</strong><br>
• Multi-format support (PDF/DOCX/TXT/URL)<br>
• Three summarization methods<br>
• Detailed evaluation metrics
</center>""", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()