# iPhone User Guide RAG Chatbot

An intelligent, Retrieval-Augmented Generation (RAG) chatbot designed to answer questions strictly based on the provided "iPhone User Guide For iOS 7.1 Software." 

Built with LangGraph, Streamlit, and Qdrant, this application utilizes a strictly guarded state machine to ensure all answers are contextually grounded, explicitly cited, and entirely free of hallucinations.

## 🛠️ Tech Stack

* **Orchestration:** LangGraph
* **UI:** Streamlit
* **Vector Database:** Qdrant Cloud
* **Package Manager:** `uv`
* **Containerization:** Docker

## 🧠 Model Configuration

As per the assessment requirements, the models used in this pipeline are free/open-tier and fully documented below:

* **Chat Model:** `gemini-3.1-flash-lite` (via Google Generative AI)
  * *Reasoning:* This model natively supports fast, reliable tool calling. It is is designed for high-volume agentic workflows. A rigid system prompt ensures the model consolidates citations and explicitly states "I cannot find the answer" if the context lacks the necessary information.
* **Embedding Model:** `all-MiniLM-L6-v2` (via HuggingFace)
  * *Reasoning:* A highly efficient, fast, and open-source embedding model that performs exceptionally well for semantic search over standard technical English text.

## 🏗️ Architecture & Engineering Decisions

The quality of retrieval is heavily dependent on how the document is processed. Even though the ingestion step is pre-computed, the methodology is detailed below:

### 1. Ingestion & Chunking Strategy
The document was loaded using `PyMuPDF4LLMLoader` in page mode. This specific loader is advantageous because it natively converts the PDF content into Markdown format.
* **Splitting Method:** `MarkdownHeaderTextSplitter`
* **Why this strategy?** Standard character or recursive splitters often slice through paragraphs arbitrarily, destroying context. By splitting on Markdown headers (`# Chapter`, `## Section`), we ensure that semantically related information stays grouped together in a single chunk.

### 2. Vector Store Metadata
During the chunking process, vital metadata is preserved and injected into each chunk.
* **Fields Stored:** `page`, `source`, `Chapter`, `Section`, `Sub_Section`
* **Why these fields?** The assessment strictly requires page and section citations for every answer. By preserving the origin file, the exact page number, and the full markdown hierarchy (from overarching Chapters down to Sub-Sections), the `search_document` RAG tool can accurately extract this metadata from Qdrant. It then prepends it to the context chunk as a literal string (e.g., `--- CONTENT SOURCE: [Page X, Chapter: Y] ---`). This guarantees the LLM has precise citation data mapped directly into its context window before generating the text.

### 3. State Management (Human-in-the-Loop)
The application utilizes LangGraph's `InMemorySaver` checkpointer to maintain conversation history (thread state) seamlessly. If the LLM generates a text response, it is interrupted and routed to a review node where the Streamlit UI captures human input, ensuring robust multi-turn conversational support.

---

## 🚀 How to Run the Application

The application is containerized and uses `uv` for lightning-fast, deterministic dependency management. Follow these exact steps to run the chatbot locally.

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

### 2. Configure Environment Variables
You will find a .env.example file in the root directory. Rename it to .env and fill in your API keys:

```bash
mv .env.example .env
```

Make sure to provide your Google API Key and your Qdrant Cluster credentials.

### 3. Build the Docker Image
Build the container using the provided Dockerfile. (Note: The Dockerfile utilizes uv sync to perfectly replicate the locked environment). Note: this build may take some time.

```bash
docker build -t chatbot:1.0 .
```

### 4. Run the Container
The application is exposed on port 8501. Run the container and map it to your local machine:

```bash
docker run -p 8501:8501 --env-file .env chatbot:1.0
```

### 5. Chat
Once the container starts running, open your web browser and navigate to:
http://localhost:8501
