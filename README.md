# RAG-Based PDF Data Retrieval Project

## Overview
This project provides a solution for retrieving and querying data from PDF files using a Retrieval-Augmented Generation (RAG) approach. Users can upload PDF files and ask questions about their content through a chatbot interface. The application allows the use of different Large Language Models (LLMs) for the chatbot, selected from a predefined list.

The frontend is built with Streamlit, and the backend leverages the power of LangChain and Ollama for processing and answering questions based on the content of the PDFs.

## Features
- **PDF Upload:** Easily upload PDF files to the application.
- **Chatbot Interface:** Ask questions about the content of the PDFs.
- **Model Flexibility:** Choose from various LLMs to process and generate responses, using models sourced via Ollama.
- **Streamlit Frontend:** Interactive and user-friendly UI built with Streamlit.
- **RAG Approach:** Combines document retrieval and generation for accurate answers.

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Setup
1. Clone this repository:
   ```bash
   git clone https://github.com/suruchiksd/RAG-PDF-Retrieval-Project.git
   cd RAG-PDF-Retrieval-Project
2. Install the Required Python Packages::
   ```bash
   pip install -r requirements.txt
3. Start the Streamlit App::
   ```bash
   streamlit run app.py

## Live Demo
[Check out the live demo]([https://your-live-demo-link.com](https://rag-based-pdf-data-retrieval-project.onrender.com))
