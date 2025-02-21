import logging.config
from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from backend.configuration import DB_INDEX, Configuration

logger = logging.getLogger(__name__)
COLLECTION_NAME = "resume_collection"


def create_or_load_vector_store(
    chunks: List[Document], vector_index_name: Path, config: Configuration
) -> VectorStore:
    """Return a vector store index on the resume
    If the path to vector store exists the vector store is loaded from there, otherwise it will be created

    Args:
        chunks: Document objects from resume
        vector_index_name: Path to vector store index
        config: Configuration object (optional)

    Returns:
        Vector store
    """
    # Store embeddings into the vector store
    vector_index_path = DB_INDEX / vector_index_name
    if not vector_index_path.exists():
        logger.info("%s: creating storage for vector index", vector_index_path)
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=config.embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=str(vector_index_path),
        )
    else:
        logger.info("%s: loading vector index", vector_index_path)
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=config.embeddings,
            persist_directory=str(vector_index_path),
        )
    if vector_store._collection.count() == 0:
        raise IOError(
            f"No Documents found in loaded vector store: {vector_index_name}, {COLLECTION_NAME}"
        )

    return vector_store
