from pathlib import Path

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.vectorstores.utils import filter_complex_metadata

from backend.configuration import PATH_CHUNKS, Configuration
from backend.pickle_utils import load_from_pickle, store_to_pickle


# Ingest the resume PDF into Documents
def create_or_load_chunks(file_path: Path, config: Configuration):
    chunks_out_path = (PATH_CHUNKS / file_path.name).with_suffix(".pkl")
    if not chunks_out_path.exists():

        loader = UnstructuredPDFLoader(file_path, mode="elements", strategy="fast")
        chunks = loader.load()
        # Filter metadata to successfully process into DB
        chunks = filter_complex_metadata(chunks)

        if not chunks_out_path.exists():
            chunks_out_path.parent.mkdir(parents=True, exist_ok=True)

        # Store the chunks
        chunks_out_path.parent.mkdir(parents=True, exist_ok=True)
        store_to_pickle(chunks, chunks_out_path)
    else:
        chunks = load_from_pickle(chunks_out_path)

    return chunks
