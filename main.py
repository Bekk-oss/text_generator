import sys
import codecs
if sys.stdout.encoding != 'UTF-8':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
if sys.stderr.encoding != 'UTF-8':
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

import os
import traceback
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from pdfminer.high_level import extract_text
import re  # For text cleaning
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

#Configuration Settings
PDF_PATH = "Fundamentals_of_Electrical_Engineering_I_9648-pages-2.pdf"
EXTRACTED_TEXT_PATH = "extracted_text.txt"

# Configure Gemini API key
# Replace with your actual Gemini API key
GEMINI_API_KEY = "key"
genai.configure(api_key=GEMINI_API_KEY)

#Text Extraction & Cleaning
def extract_and_clean_pdf(pdf_path, output_path):
    """Extracts text from PDF and cleans it for processing"""
    try:
        # Extract raw text
        raw_text = extract_text(pdf_path)

        # Clean text: Remove excessive whitespace, headers/footers
        cleaned_text = re.sub(r'\s+', ' ', raw_text)  # Replace multiple spaces
        cleaned_text = re.sub(r'-\n', '', cleaned_text)  # Fix hyphenated words

        # Save cleaned text
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(cleaned_text)

        return cleaned_text
    except Exception as e:
        raise RuntimeError(f"PDF extraction failed: {str(e)}")

#Document Processing Pipeline
try:
    # Extract and clean PDF text
    print("Extracting and cleaning PDF text...")
    cleaned_text = extract_and_clean_pdf(PDF_PATH, EXTRACTED_TEXT_PATH)

    # Load cleaned text
    loader = TextLoader(EXTRACTED_TEXT_PATH, encoding="utf-8")
    documents = loader.load()

    # Split documents into chunks
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=256,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True
    )
    texts = text_splitter.split_documents(documents)

    if not texts:
        raise ValueError("No valid text extracted after cleaning. Check PDF content.")

    # Create embeddings and vector store
    print("Creating vector database...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    vectorstore = FAISS.from_documents(texts, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})  # Retrieve 4 relevant chunks

    #LLM Initialization with Error Handling
    print("Initializing language model...")
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)
    except Exception as e:
        raise ConnectionError(f"LLM initialization failed. Check API token and model availability: {str(e)}")

    #Enhanced Prompt Engineering
    outline_template = """As an expert in integrated circuit design, create a comprehensive outline addressing:
    Question: {question}
    Context: {context}

    Structure your outline with:
    1. Key concept definitions
    2. Technical implementation approaches
    3. Performance trade-offs
    4. Real-world application examples
    Output only the outline structure."""

    draft_template = """Develop a technical response based on:
    Outline: {outline}
    Context: {context}

    Include:
    - Mathematical equations where applicable
    - Circuit diagrams described in text
    - Performance metrics (speed, power, area)
    - Citations to Razavi's design principles"""

    revise_template = """Refine this draft into publication-quality content:
    Draft: {draft}
    Context: {context}

    Improvements needed:
    1. Ensure technical accuracy against context
    2. Add transitional phrases between sections
    3. Highlight design trade-offs
    4. Use professional engineering terminology
    5. Format equations clearly: V_out = f(V_in)"""

    # Create prompt templates
    outline_prompt = PromptTemplate(input_variables=["question", "context"], template=outline_template)
    draft_prompt = PromptTemplate(input_variables=["outline", "context"], template=draft_template)
    revise_prompt = PromptTemplate(input_variables=["draft", "context"], template=revise_template)

    #Chain Construction with Memory Optimization
    outline_chain = outline_prompt | llm | StrOutputParser()
    draft_chain = draft_prompt | llm | StrOutputParser()
    revise_chain = revise_prompt | llm | StrOutputParser()

    full_chain = (
        RunnablePassthrough.assign(outline=outline_chain)
        | RunnablePassthrough.assign(draft=draft_chain)
        | revise_chain
    )

    #Enhanced Retrieval-Generation with Context Filtering
    def generate_answer(question):
        print(f"\nProcessing question: {question}")
        # Retrieve relevant context
        docs = retriever.get_relevant_documents(question)

        # Filter context by relevance score (if available)
        filtered_docs = [doc for doc in docs if doc.metadata.get('score', 1) > 0.7][:1]

        context = "\n\n---REFERENCE---\n".join([d.page_content for d in filtered_docs])
        print(f"Retrieved {len(filtered_docs)} context chunks")

        # Execute multi-step chain
        try:
            result = full_chain.invoke({"question": question, "context": context})
            return result
        except Exception as e:
            print(f"An error occurred during chain execution: {e}")
            traceback.print_exc()
            return "Error: Could not generate an answer."

    #Example Queries
    queries = [
        "Explain the differences of FFT vs DFFT" 
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n{'='*50}\nRESPONSE TO QUERY #{i}: {query}\n{'='*50}")
        answer = generate_answer(query)
        print(f"\nFINAL ANSWER:\n{answer}")
        print(f"\n{'='*50}\nEND OF RESPONSE #{i}\n{'='*50}")

except Exception as e:
    print(f"\nERROR: {str(e)}")
    traceback.print_exc()
    print("Troubleshooting Tips:")
    print("- Check PDF file exists and is readable")
    print("- Verify Hugging Face API token is valid")
    print("- Ensure sentence-transformers model is available")
    print("- Reduce chunk_size if memory errors occur")
