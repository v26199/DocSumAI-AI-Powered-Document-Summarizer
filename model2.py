from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, WebBaseLoader
from langchain_core.prompts import PromptTemplate
from groq import Groq
import os
from dotenv import load_dotenv
import pypdf
from transformers import pipeline, AutoTokenizer
import logging
from huggingface_hub import login
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest
import nltk
import tempfile
import requests
from typing import Union, BinaryIO
from bs4 import BeautifulSoup

# Download required NLTK data
nltk.download('punkt')

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables first
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables")

# Authenticate with Hugging Face using env variable
if HF_TOKEN:
    login(token=HF_TOKEN)
else:
    logging.warning("Hugging Face token not found in environment variables")

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# Initialize models and pipelines
try:
    nlp_pipeline = pipeline(
        "summarization", 
        model="facebook/bart-large-cnn",
        tokenizer="facebook/bart-large-cnn",
        device=-1  # Use CPU for stability
    )
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    nlp_extract = spacy.load('en_core_web_lg')  # Use larger model for better accuracy
except Exception as e:
    logging.error(f"Error initializing models: {e}")
    raise


def extract_text(file_or_url: Union[str, BinaryIO]) -> str:
    """
    Handle both file upload and URL input with improved error handling.
    - If input is a string starting with http(s), it's treated as a URL.
    - If input is a string (but not a URL), it's returned as-is.
    - Otherwise, if input has a .read() method, its content is read and decoded.
    """
    try:
        if isinstance(file_or_url, str):
            if file_or_url.startswith(('http://', 'https://')):
                return process_url(file_or_url)
            else:
                return file_or_url
        elif hasattr(file_or_url, "read"):
            content = file_or_url.read()
            if isinstance(content, bytes):
                return content.decode("utf-8")
            return content
        else:
            raise ValueError("Invalid input type for extract_text")
    except Exception as e:
        logging.error(f"Error extracting text: {e}")
        raise


def process_uploaded_file(uploaded_file: BinaryIO, filename: str) -> str:
    """
    Process uploaded files with proper temporary file handling.
    For text files, the content is read and decoded directly.
    For PDF and DOCX files, a temporary file is created for processing.
    """
    temp_path = None
    try:
        if filename.endswith('.txt'):
            # For plain text, read and decode directly.
            content = uploaded_file.read()
            if isinstance(content, bytes):
                return content.decode('utf-8')
            return content
        else:
            # For PDFs and DOCX files, use a temporary file.
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                content = uploaded_file.read()
                temp_file.write(content)
                temp_path = temp_file.name

            if filename.endswith('.pdf'):
                return process_pdf(temp_path)
            elif filename.endswith('.docx'):
                return process_docx(temp_path)
            else:
                raise ValueError("Unsupported file type")
    except Exception as e:
        logging.error(f"Error processing file: {e}")
        raise
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def process_pdf(file_path: str) -> str:
    """Process PDF files with error recovery."""
    try:
        pdf_file = pypdf.PdfReader(file_path)
        text = "".join(page.extract_text() for page in pdf_file.pages if page.extract_text())
        return text
    except Exception as e:
        logging.error(f"PDF processing error: {e}")
        raise


def process_docx(file_path: str) -> str:
    """Process DOCX files with proper loader."""
    try:
        return Docx2txtLoader(file_path).load()[0].page_content
    except Exception as e:
        logging.error(f"DOCX processing error: {e}")
        raise


def process_url(url: str) -> str:
    """Process URL input with improved error handling."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup.get_text()
    except Exception as e:
        logging.error(f"URL processing error: {e}")
        raise


def generate_nlp_summary(text: str, length: int) -> str:
    """Generate an abstractive summary using the NLP pipeline with input truncation and output checks."""
    try:
        # Truncate the text if it's too long for the model's context window
        truncated_text = text if len(text) < 1024 else text[:1024]
        
        # Determine output lengths (tunable parameters)
        min_output = max(int(length * 0.5), 50)
        max_output = min(length, 500)
        
        summary_output = nlp_pipeline(
            truncated_text,
            max_length=max_output,
            min_length=min_output,
            do_sample=False,
            num_beams=4
        )
        
        if summary_output and isinstance(summary_output, list) and len(summary_output) > 0:
            return summary_output[0].get('summary_text', 'No summary generated.')
        else:
            logging.error("NLP pipeline returned empty output.")
            return "Error: No summary generated."
    except Exception as e:
        logging.error(f"Error generating NLP summary: {e}")
        return f"Error generating NLP summary: {str(e)}"

def generate_extractive_summary(text: str, select_length: int) -> str:
    """Generate an extractive summary using spaCy with sentence reordering for natural flow."""
    try:
        if not text.strip():
            return "Error: Empty input text"

        doc = nlp_extract(text)
        sentences = list(doc.sents)
        
        # If there are fewer sentences than requested, return all sentences.
        if len(sentences) <= select_length:
            return " ".join(sent.text.strip() for sent in sentences)

        # Calculate normalized word frequencies
        word_frequencies = {}
        for token in doc:
            if token.text.lower() not in STOP_WORDS and token.text not in punctuation:
                lemma = token.lemma_.lower()
                word_frequencies[lemma] = word_frequencies.get(lemma, 0) + 1
        max_freq = max(word_frequencies.values()) if word_frequencies else 1
        for word in word_frequencies:
            word_frequencies[word] /= max_freq

        # Score each sentence
        sentence_scores = {}
        for sent in sentences:
            score = sum(word_frequencies.get(token.lemma_.lower(), 0) for token in sent)
            sentence_scores[sent] = score

        # Select the top sentences
        top_sentences = nlargest(select_length, sentence_scores, key=sentence_scores.get)
        # Reorder the selected sentences by their order in the document for better coherence.
        top_sentences_sorted = sorted(top_sentences, key=lambda s: s.start)
        return " ".join(sent.text.strip() for sent in top_sentences_sorted)
    except Exception as e:
        logging.error(f"Extractive summarization error: {e}")
        return f"Error generating extractive summary: {str(e)}"


def generate_llm_summary(text: str, model: str, length: int) -> str:
    """Generate summary using a Large Language Model (LLM)."""
    try:
        # Truncate text based on model's context window (e.g., 30000 tokens for mixtral)
        max_input_length = 30000 if "32768" in model else 6000
        truncated_text = text[:max_input_length]
        prompt_template = "Summarize the following text in {} words: {}"
        
        response = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": prompt_template.format(length, truncated_text)
            }],
            temperature=0.5,
            max_tokens=min(length * 2, 4000),
            top_p=1.0,
            frequency_penalty=0.2,
            presence_penalty=0.2
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"LLM summarization error: {e}")
        return f"Error generating LLM summary: {str(e)}"
