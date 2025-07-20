# Advanced RAG Text Generator

This project implements a sophisticated Retrieval-Augmented Generation (RAG) pipeline using LangChain, FastAPI, and Google's Gemini Pro model. The system is designed to answer complex questions based on a provided PDF document, simulating a real-world scenario where a model needs to be augmented with specific, external knowledge.

The application is containerized with Docker and designed for easy deployment on Google Cloud, making it a robust and scalable solution for advanced text generation tasks. It also includes a complete MLOps pipeline for side-by-side (AutoSxS) model evaluation using Vertex AI Pipelines.

## Features

- **Retrieval-Augmented Generation (RAG):** Ingests a PDF document, creates a vector index of its content, and uses it to provide context to the language model for more accurate and relevant answers.
- **Multi-Stage Generation Chain:** Implements a sophisticated `Outline -> Draft -> Revise` chain using LangChain Expression Language (LCEL) for high-quality, structured output.
- **FastAPI-based API:** Exposes the RAG pipeline via a robust and modern REST API, including a health check and prediction endpoint.
- **Google Cloud Native:**
    - Securely fetches the Gemini API key from **Google Secret Manager**.
    - Downloads the source PDF document from **Google Cloud Storage (GCS)**.
    - Designed for easy deployment to **Google Cloud Run**.
- **Dockerized:** Comes with a `Dockerfile` for consistent, reproducible builds and deployments.
- **Vertex AI Evaluation:** Includes a `autosxs_pipeline.py` script to run automated side-by-side evaluations comparing the RAG model against a baseline (Gemini 1.5 Flash).

## Project Structure

```
.
├── main.py                   # FastAPI application with the RAG logic
├── Sample_textgenerator(2).ipynb #the google colab version for simple access
├── Dockerfile                # Docker configuration for containerizing the app
├── requirements.txt          # Python dependencies
├── autosxs_pipeline.py       # Vertex AI pipeline for model evaluation (not provided)
├── generate_predictions.py   # Script to generate predictions for evaluation
├── run_evaluation.py         # Script to execute the evaluation pipeline
└── 
```

## Installation and Local Setup

These instructions are for running the application on your local machine for development and testing.

### Prerequisites

- Python 3.9+
- `pip` for package management
- A Google Cloud Project

### Steps

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd textgenerator
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up Google Cloud Authentication:**
    You need to be authenticated to access Google Cloud Storage and Secret Manager.
    ```bash
    gcloud auth application-default login
    ```

4.  **Configure Environment Variables:**
    The application requires the following environment variables. You can set them in your shell or create a `.env` file.
    ```bash
    export PROJECT_ID="your-gcp-project-id"
    export GCS_PDF_PATH="gs://your-bucket-name/path/to/your/document.pdf"
    ```
    - `PROJECT_ID`: Your Google Cloud Project ID.
    - `GCS_PDF_PATH`: The path to the PDF file in a GCS bucket that the application will use as its knowledge base.

5.  **Store API Key in Secret Manager:**
    The application securely fetches your Gemini API key from Google Secret Manager.
    ```bash
    # 1. Enable the Secret Manager API
    gcloud services enable secretmanager.googleapis.com

    # 2. Create a secret to hold the key
    gcloud secrets create gemini-api-key --replication-policy="automatic"

    # 3. Add your API key as a secret version
    printf "YOUR_GEMINI_API_KEY" | gcloud secrets versions add gemini-api-key --data-file=-
    ```

6.  **Run the application:**
    The application uses `uvicorn` to run the FastAPI server.
    ```bash
    uvicorn main:app --reload
    ```
    The server will be running at `http://127.0.0.1:8000`.

## Docker Usage

Containerizing the application with Docker is the recommended way to ensure consistency between development and production environments.

### Prerequisites

- Docker installed and running.

### Steps

1.  **Build the Docker Image:**
    From the project root directory, run the build command:
    ```bash
    docker build -t text-generator-app .
    ```

2.  **Run the Docker Container:**
    You need to pass the environment variables and Google Cloud credentials to the container.
    ```bash
    docker run -p 8080:8080 \
      -e PROJECT_ID="your-gcp-project-id" \
      -e GCS_PDF_PATH="gs://your-bucket-name/path/to/your/document.pdf" \
      --mount type=bind,source="$HOME/.config/gcloud",target=/root/.config/gcloud,readonly \
      text-generator-app
    ```
    - The `-p 8080:8080` flag maps the container's port 8080 to your local machine's port 8080.
    - The `--mount` command securely shares your local Google Cloud credentials with the container.

## Deployment to Google Cloud Run 

Google Cloud Run is the ideal platform for deploying this application for personal use or demonstrations, and it also provides a public HTTPS URL. (VertexAI is also an option)

### Prerequisites

- A Google Cloud Project with billing enabled.
- `gcloud` CLI installed and configured.
- Your Docker image pushed to Google Artifact Registry.

### Steps

1.  **Enable APIs:**
    ```bash
    gcloud services enable run.googleapis.com
    gcloud services enable artifactregistry.googleapis.com
    ```

2.  **Create an Artifact Registry Repository:**
    This is where you'll store your Docker image.
    ```bash
    gcloud artifacts repositories create my-text-generator-repo \
      --repository-format=docker \
      --location=us-central1
    ```

3.  **Configure Docker Authentication:**
    ```bash
    gcloud auth configure-docker us-central1-docker.pkg.dev
    ```

4.  **Tag and Push the Docker Image:**
    Replace `your-gcp-project-id` with your actual project ID.
    ```bash
    # Tag the image
    docker tag text-generator-app us-central1-docker.pkg.dev/your-gcp-project-id/my-text-generator-repo/text-generator-app

    # Push the image
    docker push us-central1-docker.pkg.dev/your-gcp-project-id/my-text-generator-repo/text-generator-app
    ```

5.  **Deploy to Cloud Run:**
    This single command deploys your container and sets up all the necessary configurations.
    ```bash
    gcloud run deploy text-generator-service \
      --image=us-central1-docker.pkg.dev/your-gcp-project-id/my-text-generator-repo/text-generator-app \
      --platform=managed \
      --region=us-central1 \
      --allow-unauthenticated \
      --set-env-vars="PROJECT_ID=your-gcp-project-id,GCS_PDF_PATH=gs://your-bucket-name/path/to/your/document.pdf" \
      --update-secrets=GEMINI_API_KEY=gemini-api-key:latest
    ```
    - `--allow-unauthenticated`: Makes the service publicly accessible.
    - `--set-env-vars`: Sets the required environment variables.
    - `--update-secrets`: Securely injects the Gemini API key from Secret Manager into the application.

After the command completes, it will provide you with a public **Service URL**. You can now send POST requests to the `/predict` endpoint on that URL.
