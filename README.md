# RAG-based Question Answering App

This repository contains a Streamlit application for RAG-based Question Answering (QA). Users can upload a PDF/txt document and ask a question. The application will analyze the document to retrieve the most relevant sections and generate an answer.

## Features
- Upload PDF/txt files
- Ask questions about the document content
- Retrieves relevant document sections based on embeddings
- Generates answers using LLM

## Installation

1. **Clone this repository**:
    ```bash
    git clone https://github.com/yourusername/your-repo-name.git
    cd your-repo-name
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Set up OpenAI API key**:
   Create a `.env` file in the root directory with your OpenAI API key:
   ```plaintext
   OPENAI_API_KEY=your_openai_api_key
