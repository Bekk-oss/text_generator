import sys
import codecs
if sys.stdout.encoding != 'UTF-8':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
if sys.stderr.encoding != 'UTF-8':
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

import os
import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from pdfminer.high_level import extract_text
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from google.cloud import storage, secretmanager

# --- Global Variables & Configuration ---

# Will be populated by the startup event
llm = None
retriever = None
full_chain = None

# --- FastAPI Lifespan Management ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    # This code runs on startup
    print("INFO:     Lifespan startup event...")
    global llm, retriever, full_chain

    try:
        # --- Configuration Settings ---
        GCS_PDF_PATH = os.environ.get("GCS_PDF_PATH")
        PROJECT_ID = os.environ.get("PROJECT_ID")
        LOCAL_PDF_PATH = "/tmp/downloaded.pdf"
        EXTRACTED_TEXT_PATH = "/tmp/extracted_text.txt"

        if not all([GCS_PDF_PATH, PROJECT_ID]):
            raise ValueError("Missing required environment variables: GCS_PDF_PATH, PROJECT_ID")

        # --- Secret Manager ---
        print("INFO:     Fetching API key from Secret Manager...")
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{PROJECT_ID}/secrets/gemini-api-key/versions/latest"
        response = client.access_secret_version(name=name)
        GEMINI_API_KEY = response.payload.data.decode('UTF-8')
        genai.configure(api_key=GEMINI_API_KEY)
        print("INFO:     Successfully configured Gemini API key.")

        # --- GCS File Handling ---
        print(f"INFO:     Downloading PDF from {GCS_PDF_PATH}...")
        storage_client = storage.Client()
        bucket_name, blob_name = GCS_PDF_PATH.replace("gs://", "").split("/", 1)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.download_to_filename(LOCAL_PDF_PATH)
        print(f"INFO:     Successfully downloaded PDF to {LOCAL_PDF_PATH}")

        # --- Text Extraction & Cleaning ---
        print("INFO:     Extracting and cleaning PDF text...")
        raw_text = extract_text(LOCAL_PDF_PATH)
        cleaned_text = re.sub(r'\s+', ' ', raw_text)
        cleaned_text = re.sub(r'-\n', '', cleaned_text)
        with open(EXTRACTED_TEXT_PATH, "w", encoding="utf-8") as f:
            f.write(cleaned_text)
        print("INFO:     Successfully extracted and cleaned text.")

        # --- Document Processing ---
        print("INFO:     Loading and splitting documents...")
        loader = TextLoader(EXTRACTED_TEXT_PATH, encoding="utf-8")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=256, chunk_overlap=200, length_function=len, add_start_index=True
        )
        texts = text_splitter.split_documents(documents)
        if not texts:
            raise ValueError("No text chunks created after splitting. Check PDF content.")
        print(f"INFO:     Created {len(texts)} text chunks.")

        # --- Vector Store & Retriever ---
        print("INFO:     Creating vector database...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        vectorstore = FAISS.from_documents(texts, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        print("INFO:     Vector database created successfully.")

        # --- LLM & Chain Initialization ---
        print("INFO:     Initializing language model...")
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)

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

        outline_prompt = PromptTemplate(input_variables=["question", "context"], template=outline_template)
        draft_prompt = PromptTemplate(input_variables=["outline", "context"], template=draft_template)
        revise_prompt = PromptTemplate(input_variables=["draft", "context"], template=revise_template)

        outline_chain = outline_prompt | llm | StrOutputParser()
        draft_chain = draft_prompt | llm | StrOutputParser()
        revise_chain = revise_prompt | llm | StrOutputParser()

        full_chain = (
            RunnablePassthrough.assign(outline=outline_chain)
            | RunnablePassthrough.assign(draft=draft_chain)
            | revise_chain
        )
        print("INFO:     Full generation chain created.")
        print("INFO:     Startup complete. Application is ready.")

    except Exception as e:
        print(f"FATAL:    Startup failed: {str(e)}")
        traceback.print_exc()
        # In a real scenario, you might want to exit or prevent the app from serving requests

    yield
    # This code runs on shutdown
    print("INFO:     Lifespan shutdown event...")

# --- FastAPI App Definition ---

app = FastAPI(lifespan=lifespan)

class Instance(BaseModel):
    question: str

class PredictionPayload(BaseModel):
    instances: list[Instance]

@app.get("/health", status_code=200)
async def health_check():
    """Vertex AI health check. Confirms the server is running."""
    return {"status": "healthy"}

@app.post("/predict")
async def predict(payload: PredictionPayload):
    """Vertex AI prediction endpoint."""
    if not full_chain or not retriever:
        raise HTTPException(status_code=503, detail="Model is not ready. Check startup logs.")

    try:
        answers = []
        for instance in payload.instances:
            docs = retriever.get_relevant_documents(instance.question)
            filtered_docs = [doc for doc in docs if doc.metadata.get('score', 1) > 0.7][:1]
            context = "\n\n---\n".join([d.page_content for d in filtered_docs])
            result = full_chain.invoke({"question": instance.question, "context": context})
            answers.append(result)
        return {"predictions": answers}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")