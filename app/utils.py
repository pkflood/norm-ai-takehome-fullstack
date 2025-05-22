from pydantic import BaseModel
import qdrant_client
import pypdf
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





