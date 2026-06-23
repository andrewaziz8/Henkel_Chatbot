import os
import pprint
from uuid import uuid4
from dotenv import load_dotenv
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

# Load the variables from the .env file into the environment
load_dotenv()
api_key_google = os.environ.get("GOOGLE_API_KEY")
qdrant_url = os.environ.get("QDRANT_URL")
api_key_qdrant = os.environ.get("QDRANT_API_KEY")

file_path = "iPhone User Guide 1.pdf"

# 1. PDF Document Loader
loader = PyMuPDF4LLMLoader(file_path, mode="page",)
docs = loader.load()
print(len(docs))
pprint.pp(docs[7].metadata)
print(docs[7])

# 2. Split the document into sections based on headers
headers_to_split_on = [
    ("#", "Chapter"),
    ("##", "Section"),
    ("###", "Sub_Section"),
]

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on, strip_headers=False) # Keep the header text in the chunk for LLM context

final_chunks = []

# Iterate through each page to extract sections while keeping the page number
for page_doc in docs:
    # This splits the page's markdown into smaller chunks based on the headers
    # and automatically adds {"Chapter": "...", "Section": "..."} to the metadata
    section_chunks = markdown_splitter.split_text(page_doc.page_content)
    
    # Re-attach the page number to these newly created section chunks
    for chunk in section_chunks:
        # PyMuPDF usually 0-indexes pages, so add 1 for human readability
        current_page = page_doc.metadata.get("page", 0) + 1 
        
        # Inject the original page and source into the new chunk's metadata
        chunk.metadata["page"] = current_page
        chunk.metadata["source"] = file_path
        
        final_chunks.append(chunk)

print(f"Total chunks created: {len(final_chunks)}")
print("\nsample chunk:")
pprint.pp(final_chunks[24])

# 3. Create the Embedding
# embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview", api_key=api_key_google)
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
# embeddings = HuggingFaceEmbeddings(model_name="Qwen/Qwen3-Embedding-0.6B")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 4. Create the vector Store using Qdrant
qdrant = QdrantVectorStore.from_documents(
    final_chunks,
    embeddings,
    url=qdrant_url,
    prefer_grpc=True,
    force_recreate=True,  # Set to True to overwrite the existing collection if it exists
    api_key=api_key_qdrant,
    collection_name="iphone_user_guide",
)

# # Connect to the already-populated database
# qdrant = QdrantVectorStore.from_existing_collection(
#     collection_name="iphone_user_guide",
#     embedding=embeddings, # The same embedding model you used in ingest.py
#     url=qdrant_url,
#     prefer_grpc=True,
#     api_key=api_key_qdrant,
# )

# to add documents to the vector store, you can use the following command:
# uuids = [str(uuid4()) for _ in range(len(final_chunks))]
# qdrant.add_documents(documents=final_chunks, ids=uuids)

# to delete from the vector store, you can use the following command:
# qdrant.delete(ids=[uuids[-1]])

results = qdrant.similarity_search(
    "Are there any virtual buttons on the touchscreen", k=5
)
for res in results:
    print(f"* {res.page_content} [{res.metadata}]")