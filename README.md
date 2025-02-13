# DocSumAI: AI-Powered Document Summarizer 📚

DocSumAI is an advanced AI-powered tool designed to summarize documents efficiently and accurately. It supports multiple file formats (PDF, DOCX, TXT) and URLs, offering three distinct summarization methods: **LLM-based**, **NLP pipeline-based**, and **extractive summarization**. The tool also provides detailed evaluation metrics to assess the quality of the generated summaries.

---

## Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Evaluation Metrics](#evaluation-metrics)
5. [Dependencies](#dependencies)
6. [Repository Link](#repository-link)
7. [Contributing](#contributing)
8. [License](#license)
9. [Contact](#contact)

---

## Features ✨

- **Multi-format Support**: Summarize documents in PDF, DOCX, TXT formats, or directly from URLs.
- **Three Summarization Methods**:
  - **LLM Summarization**: Leverages advanced language models like `llama3-70b-8192` for high-quality summaries.
  - **NLP Pipeline Summarization**: Uses the `facebook/bart-large-cnn` model for abstractive summarization.
  - **Extractive Text Summarization**: Extracts key sentences using spaCy's NLP capabilities.
- **Evaluation Metrics**: Provides BLEU, ROUGE-1, ROUGE-2, ROUGE-L, and METEOR scores to evaluate summary quality.
- **Visualization**: Generates bar charts to represent evaluation metrics visually.
- **Scalability**: Built with modularity and scalability in mind, making it easy to extend or integrate into larger systems.

---

## Installation

### Prerequisites
- Python 3.8+
- Install dependencies using `requirements.txt`.

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/v26199/DocSumAI-AI-Powered-Document-Summarizer.git
   cd DocSumAI-AI-Powered-Document-Summarizer
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Create a `.env` file in the root directory and add your API keys:
     ```
     HF_TOKEN=your_huggingface_token
     GROQ_API_KEY=your_groq_api_key
     ```
   - **Important**: Do not commit this `.env` file to your repository. Add it to `.gitignore`.

4. Run the app:
   ```bash
   streamlit run app.py
   ```

---

## Usage

1. **Upload or Enter URL**:
   - Choose to upload a file (PDF, DOCX, TXT) or enter a URL to summarize web content.
   - Select the desired summarization method and configure settings like summary length or number of extractive sentences.

2. **View Results**:
   - The app generates a summary and displays evaluation metrics to help you understand the quality of the output.

---

## Evaluation Metrics

The app evaluates summaries using the following metrics:
- **BLEU**: Measures n-gram overlap between the summary and the original text.
- **ROUGE-1, ROUGE-2, ROUGE-L**: Assess unigram recall, bigram recall, and longest common subsequence, respectively.
- **METEOR**: Considers synonymy and alignment between the generated and reference texts.

A bar chart is generated to visually represent these metrics, helping users quickly interpret the quality of the generated summary.

---

## Dependencies

- `streamlit`: For building the interactive web app.
- `langchain`: For document loading and processing.
- `transformers`: For abstractive summarization using Hugging Face models.
- `spacy`: For extractive summarization and NLP tasks.
- `groq`: For LLM-based summarization.
- `rouge-score`: For calculating ROUGE metrics.
- `nltk`: For tokenization and BLEU/METEOR calculations.
- `requests` and `BeautifulSoup`: For web scraping.

---

## Repository Link

You can access the full source code and documentation for this project on GitHub:

[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue?style=for-the-badge&logo=github)](https://github.com/v26199/DocSumAI-AI-Powered-Document-Summarizer)

Alternatively, visit the repository directly:
- [DocSumAI GitHub Repository](https://github.com/v26199/DocSumAI-AI-Powered-Document-Summarizer)

---

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests to improve this project. Here’s how you can contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeatureName`).
3. Commit your changes (`git commit -m "Add YourFeatureName"`).
4. Push to the branch (`git push origin feature/YourFeatureName`).
5. Open a pull request.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contact

For questions or feedback, feel free to reach out:

- Email: vishubpatel4@gmail.com
- GitHub: [@v26199](https://github.com/v26199)


---
