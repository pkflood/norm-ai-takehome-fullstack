import json

from pydantic import BaseModel
import qdrant_client
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
import re
import llama_index

from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document
from llama_index.core.node_parser import SentenceSplitter, SimpleNodeParser
from typing import List

import fitz



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
    """
    DocumentService that creates exactly 11 documents - one for each law.
    Uses LlamaIndex for PDF parsing with custom law-specific splitting.
    """

    def create_documents(self, file_path: str = None) -> List[Document]:
        """
        Extract pdf text and load contents into documents for each primary law

        """
        if not file_path:
            raise ValueError("No file path provided")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            # # Load the PDF and get the raw text
            print(f"📖 Loading PDF: {file_path}")
            # pages = SimpleDirectoryReader(input_files=[file_path]).load_data()
            #
            #
            # # Combine all pages into one text block
            # full_text = ""
            # for page in pages:
            #     full_text += page.text + "\n"
            # print(full_text)

            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text("text")  # use "text" mode to retain layout
            print("testing new pdf package")
            print(text)

            # Split by law sections to create one document per major law
            print("✂️ Splitting into law sections...")
            laws = self._split_into_laws(text)

            laws.pop(0) #remove title of page from list of laws

            documents = []

            # Creating document with metadata={"Section": "Law #"} and text is law text for each major number law
            # Possible improvement with more time would be to correct the spacing and general format for the text; was tricky because pdf parsing removes formatting and spaces.
            # Could do this either with advanced regex/parsing functions or llm call on the text following extraction.
            print(len(laws))
            for index, law in enumerate(laws):
                law = law.replace('\n', ' ')
                metadata = {"Section": f"Law {index + 1}"}
                document = Document(metadata=metadata, text=law)
                print(document)
                documents.append(document)

            print(f"✅ Successfully created {len(documents)} law documents")
            # return law_documents
            return documents

        except Exception as e:
            print(f"❌ Error processing PDF: {e}")
            return []


    def _split_into_laws(self, text: str) -> List[Document]:
        """
        Split the text into exactly sections for each main law
        Assuming laws will be listed in a numeric list with a digit followed by a colon.
        """
        # Split by main numbered sections (1., 2., 3., etc.)
        # This way response mentions the major law number and the text includes all subsequent parts of that law. Could also configure to provide the exact law subpoint
        # as the citation but that might be too granular for some cases and would exclude some crucial other parts of the general law depending on k value.

        # old regex process for llama parser
        #pattern = r'(?<!\d\.)\d+\. '
        # Remove empty sections and clean up
        #laws = [law.strip() for law in laws if law.strip()]

        #new process for new parser to preserve spacing in citation
        pattern = r'(?<=\n)(\d+)\.\n'
        laws = re.split(pattern, text)

        #size is number of laws (laws is originally twice the length of # laws because one law element for number, then also one for text)
        size = len(laws)//2

        #delete just number law elements (digit length will always be less than or equal to
        laws = [law.strip() for law in laws if len(law) > len(str(abs(size)))]

        return laws


# Vector Store for document embeddings and retrieval
# connect method sets up an in memory qdrant vector store, using openai gpt4 for the embedding model
# load adds docs to the vector store
# k is the number of vectors to return based on semantic similarity
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

    def load(self, docs=list[Document]):
        self.index.insert_nodes(docs)

    def query(self, query_str: str) -> Output:
        """
        This method needs to initialize the query engine, run the query, and return
        the result as a pydantic Output class. This is what will be returned as
        JSON via the FastAPI endpoint. Feel free to do this however you'd like, but
        its worth noting that the llama-index package has a CitationQueryEngine...

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

    doc_service = DocumentService()  # implemented
    docs = doc_service.create_documents()  # implemented

    index = QdrantService()  # implemented
    index.connect()  # implemented
    index.load()  # implemented

    index.query("what happens if I steal?")  # implemented