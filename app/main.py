from fastapi import FastAPI, Query
from app.utils import Output, DocumentService, QdrantService, Input
import os

app = FastAPI()

"""
Please create an endpoint that accepts a query string, e.g., "what happens if I steal 
from the Sept?" and returns a JSON response serialized from the Pydantic Output class.
"""

# Initialize services
doc_service = DocumentService()
qdrant_service = QdrantService(k=2)  # Get top 2 most relevant results

# Path to PDF document
PDF_PATH = "docs/laws.pdf"


# Initialize the vector store on startup
@app.on_event("startup")
async def startup():
    # Connect to the vector store
    qdrant_service.connect()

    # Check if the PDF file exists
    if os.path.exists(PDF_PATH):
        # Load and process the document
        docs = doc_service.create_documents(PDF_PATH)
        # Add documents to the vector index
        qdrant_service.load(docs)
    else:
        print(f"Warning: PDF file not found at {PDF_PATH}")

@app.get("/query")
async def query(query: str = Query(..., description="The query string")):
    # Process query through the QdrantService
    output = qdrant_service.query(query)
    return output