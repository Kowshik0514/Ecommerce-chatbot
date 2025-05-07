# Customer Support Chatbot

A RAG-based customer support chatbot for Electronics using Streamlit, LangChain, ChromaDB, and Gemini API.

## Features

* Conversational Q\&A based on product knowledge and FAQ JSON
* Gemini API as the LLM backend
* LangChain for retrieval and memory
* Chroma vector store for document embedding search
* Simple Streamlit UI

## Installation

Clone the repo and install dependencies:

```bash
pip install -r requirements.txt
```

## 🔑 API Key Setup

Set your Gemini API key as an environment variable:

### On macOS/Linux:

```bash
export GEMINI_API_KEY=your-gemini-api-key
```

### On Windows (CMD):

```cmd
set GEMINI_API_KEY=your-gemini-api-key
```

## data Files

* `faq.json`: JSON file containing questions and answers
* `product_knowledge.txt`: Automatically generated product data

## Running the App

```bash
streamlit run app.py
```

Then open the app in your browser.

## Project Structure

```
├── faq.json
├── knowledge_loader.py
├── main.py
├── product_knowledge.txt
├── requirements.txt
└── .env
```
