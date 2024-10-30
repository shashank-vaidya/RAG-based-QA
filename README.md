# RAG-based QA App with ChromaDB

A robust Retrieval-Augmented Generation (RAG) system that leverages **ChromaDB**, **Streamlit**, **OpenAI's GPT-4o**, and **Sentence Transformers** to enable document-level question-answering from PDFs. This system processes document chunks, vectorizes them with embeddings, and integrates dynamic similarity and keyword-based retrieval for highly accurate responses.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Project Architecture](#project-architecture)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Customization](#customization)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview

This project enables users to upload a PDF document and query its contents with natural language questions. The RAG-based approach combines keyword filtering with dynamic embedding similarity to retrieve relevant text chunks and use GPT-4o to generate accurate answers. Results are displayed in a user-friendly **Streamlit** interface, with relevant document sections highlighted for further context.

---

## Features

- **Dynamic Embedding Similarity**: Uses percentile-based thresholds for relevance, ensuring contextually accurate retrieval.
- **Keyword-Driven Retrieval**: Extracts keywords from questions, enabling relevant chunk filtering before embedding similarity.
- **Chunked Paragraph Processing**: Efficiently divides PDFs into chunks, with overlapping word sections for optimal context retention.
- **GPT-4o Answer Generation**: Provides coherent, context-based responses from extracted document sections.
- **Real-time Interface with Streamlit**: User-friendly interface for question input and answer display.
- **Document Context Display**: Shows relevant document sections alongside answers for transparency and in-depth understanding.
  
---

## Project Architecture

1. **Chunking & Preprocessing**: PDF text is split into overlapping chunks. Each chunk is cleaned (stopword removal, lowercasing) and stored with its original version.
2. **Embedding & Storage**: Cleaned chunks are vectorized with Sentence Transformers and stored in ChromaDB, enabling efficient embedding similarity searches.
3. **Keyword Extraction & Filtering**: Keywords are extracted from each question and used to filter document chunks, focusing only on contextually relevant sections.
4. **Similarity-Based Retrieval**: Dynamic similarity thresholds ensure only the most contextually relevant chunks are retrieved.
5. **Answer Generation**: GPT-4o processes retrieved chunks to generate a cohesive answer, providing contextually supported responses.

---

## Setup and Installation

### Prerequisites

- Python 3.8 or later
- OpenAI API Key (for GPT-4o usage)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt

