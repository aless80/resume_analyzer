from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document

from backend.configuration import DB_INDEX, Configuration

COLLECTION_NAME = "resume_collection"

config = Configuration()


def create_or_load_vector_store(chunks: List[Document], vector_index_name: Path):
    # Store embeddings into the vector store
    vector_index_path = DB_INDEX / vector_index_name
    if not vector_index_path.exists():
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=config.embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=str(vector_index_path),
        )
    else:
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=config.embeddings,
            persist_directory=str(vector_index_path),
        )
    if vector_store._collection.count() == 0:
        raise f"No Documents found in loaded vector store: {vector_index_name}, {COLLECTION_NAME}"

    return vector_store
