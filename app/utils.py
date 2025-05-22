from pydantic import BaseModel
import qdrant_client
import pypdf
import re
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.schema import Document
from llama_index.core import Settings
from llama_index.core import (
    VectorStoreIndex,
    ServiceContext,
)
from dataclasses import dataclass
import os

key = os.environ['OPENAI_API_KEY']

@dataclass
class Input:
    query: str
    file_path: str

@dataclass
class Citation:
    source: str
    text: str

class Output(BaseModel):
    query: str
    response: str
    citations: list[Citation]

class DocumentService:

    """
    Update this service to load the pdf and extract its contents.
    The example code below will help with the data structured required
    when using the QdrantService.load() method below. Note: for this
    exercise, ignore the subtle difference between llama-index's 
    Document and Node classes (i.e, treat them as interchangeable).

    # example code
    def create_documents() -> list[Document]:

        docs = [
            Document(
                metadata={"Section": "Law 1"},
                text="Theft is punishable by hanging",
            ),
            Document(
                metadata={"Section": "Law 2"},
                text="Tax evasion is punishable by banishment.",
            ),
        ]

        return docs

     """

    def create_documents(self, file_path: str) -> list[Document]:
        docs = []

        # Open and read the PDF file
        with open(file_path, "rb") as file:
            pdf_reader = pypdf.PdfReader(file)

            # First, extract all text from the PDF
            full_text = ""
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"

            # Define heading patterns
            heading_patterns = [
                '''
                r"(?m)^#+\s+(.+)$",                   # Markdown headings
                r"(?m)^Chapter\s+\d+[.:]\s*(.+)$",    # Chapter headings
                r"(?m)^Section\s+[\d\.]+[.:]\s*(.+)$", # Section headings
                r"(?m)^[A-Z][A-Z\s]{3,30}$",          # ALL CAPS headings (likely titles)
                r"(?m)^[IVX]+\.\s+(.+)$",             # Roman numeral headings

                '''

                r"(?m)^\d+\.\s+(.+)$",  # Simple numbered lists (e.g., 1. Introduction)
                r"(?m)^\d+\.\d+\s+(.+)$",  # Decimal numbered points (e.g., 1.1, 2.3)
                r"(?m)^(\d+\.){2,}\d+\s+(.+)$"  # Multi-level numbering (e.g., 1.1.2, 2.3.4.5)
            ]

            # Combine all patterns
            combined_pattern = "|".join(f"({pattern})" for pattern in heading_patterns)

            # using python re.split() to split the text into sections based on the heading patterns
            sections = re.split(combined_pattern, full_text)

            if len(sections) <= 1:
                # No headings found, fall back to page-based splitting
                print("No headings found")
                '''

                for i, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    if not text.strip():
                        continue

                    doc = Document(
                        metadata={"Section": f"Page {i+1}"},
                        text=text
                    )
                    docs.append(doc)
                '''
            else:
                # Process sections based on headings
                current_heading = "Introduction"
                current_content = ""

                # Process each section
                for i, section in enumerate(sections):
                    if not section or section.strip() == "":
                        continue

                    # Check if this section is a heading
                    is_heading = False
                    for pattern in heading_patterns:
                        if re.match(pattern, section.strip()):
                            is_heading = True
                            break

                    if is_heading:
                        # Save previous section if it exists
                        if current_content.strip():
                            doc = Document(
                                metadata={"Section": current_heading},
                                text=current_content.strip()
                            )
                            docs.append(doc)

                        # Start new section
                        current_heading = section.strip()
                        current_content = ""
                    else:
                        # Add to current content
                        current_content += section

                # Add the last section
                if current_content.strip():
                    doc = Document(
                        metadata={"Section": current_heading},
                        text=current_content.strip()
                    )
                    docs.append(doc)

        # If no documents were created, create a single document with all content
        if not docs and full_text.strip():
            doc = Document(
                metadata={"Section": "Full Document"},
                text=full_text.strip()
            )
            docs.append(doc)

        return docs

#Vector Store for document embeddings and retrieval
#connect method sets up an in memory qdrant vector store, using openai gpt4 for the embedding model
#load adds docs to the vector store
#k is the number of vectors to return based on semantic similarity
class QdrantService:
    def __init__(self, k: int = 2):
        self.index = None
        self.k = k
    
    def connect(self) -> None:
        client = qdrant_client.QdrantClient(location=":memory:")
                
        vstore = QdrantVectorStore(client=client, collection_name='temp')

        Settings.llm = OpenAI(api_key=key, model="gpt-4")
        Settings.embed_model = OpenAIEmbedding()

        self.index = VectorStoreIndex.from_vector_store(
            vector_store=vstore,
            embed_model=Settings.embed_model
        )

    def load(self, docs = list[Document]):
        self.index.insert_nodes(docs)
    
    def query(self, query_str: str) -> Output:

        """
        This method needs to initialize the query engine, run the query, and return
        the result as a pydantic Output class. This is what will be returned as
        JSON via the FastAPI endpount. Fee free to do this however you'd like, but
        a its worth noting that the llama-index package has a CitationQueryEngine...

        Also, be sure to make use of self.k (the number of vectors to return based
        on semantic similarity).

        # Example output object
        citations = [
            Citation(source="Law 1", text="Theft is punishable by hanging"),
            Citation(source="Law 2", text="Tax evasion is punishable by banishment."),
        ]

        output = Output(
            query=query_str, 
            response=response_text, 
            citations=citations
            )
        
        return output

        """
        from llama_index.core.query_engine import CitationQueryEngine

        # initialize the citation query engine with top k similarity results
        query_engine = CitationQueryEngine.from_args(
            self.index,
            similarity_top_k=self.k,
            # Include citations in the response, am assuming standard chunk size of 1024 but can adjust for more precision.
            citation_chunk_size=1024
        )

        # Get query response
        response = query_engine.query(query_str)

        # extract citations from the response
        source_nodes = response.source_nodes
        citations = []
        for node in source_nodes:
            source = node.metadata.get("Section", "")
            text = node.text
            citations.append(Citation(source=source, text=text))

        # create output
        output = Output(
            query=query_str,
            response=str(response),
            citations=citations
        )

        return output
       

if __name__ == "__main__":
    # Example workflow
    doc_serivce = DocumentService() # implemented
    docs = doc_serivce.create_documents() # NOT implemented

    index = QdrantService() # implemented
    index.connect() # implemented
    index.load() # implemented

    index.query("what happens if I steal?") # NOT implemented





