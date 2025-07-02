# PDF-based RAG Assistant with Gemini



A Retrieval-Augmented Generation (RAG) system that answers questions using content from PDF documents. Implements multi-step reasoning with Google Gemini for technical domain expertise.

## Features

- 📄 PDF text extraction and cleaning
- 🔍 Semantic search with FAISS vector database
- 🧠 3-stage reasoning (Outline → Draft → Revise)
- ⚡ Google Gemini 1.5 Flash for efficient generation
- 📚 Domain-specific technical Q&A capabilities

## How RAG Works in This Implementation

Retrieval-Augmented Generation combines information retrieval with language models:

1. **Indexing Phase**:
   - PDF text is extracted and cleaned
   - Content is split into overlapping chunks
   - Chunks are converted to embeddings using MPNet model
   - Embeddings stored in FAISS vector database

2. **Query Phase**:
   - User question is received
   - Relevant document chunks are retrieved
   - Top chunks are combined into context string

3. **Generation Phase**:
   - **Outline Stage**: Creates response structure
   - **Draft Stage**: Expands with technical details
   - **Revise Stage**: Polishes for professional quality
   - Gemini generates responses using retrieved context
   - 
### Key RAG Implementation Features

1. **Contextual Retrieval**:
   - Uses FAISS vector store for efficient similarity search
   - MPNet embeddings capture semantic meaning
   - Overlapping chunks preserve context

2. **Multi-Stage Reasoning**:
   ```mermaid
   graph TD
   A[Question] --> B(Outline)
   B --> C(Draft)
   C --> D(Revise)
   D --> E[Polished Answer]
## Requirements

- Python 3.10+
- Google Gemini API key

## Installation
just need the required libraries: 

pip install langchain
langchain-community
langchain-google-genai
langchain-text-splitters
pdfminer.six
faiss-cpu
sentence-transformers
google-generativeai

configure your specific pdf to the path. PDF_PATH = "path/to/your/document.pdf"



