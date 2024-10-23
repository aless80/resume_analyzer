from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.configuration import Configuration


# Load and split the PDF document and return the documents and text chunks
def load_split_pdf(file_path, config: Configuration):
    # Load the PDF document and split it into chunks
    loader = PyPDFLoader(file_path)  # Initialize the PDF loader with the file path
    documents = loader.load()  # Load the PDF document

    # Initialize the recursive character text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            " ",
            "",
        ],  # Define resume-specific separators for splitting
    )

    # Split the loaded documents into chunks
    chunks = text_splitter.split_documents(documents)
    return documents, chunks
