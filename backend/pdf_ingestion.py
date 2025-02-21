import logging
from pathlib import Path
from typing import List

from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document

from backend.configuration import PATH_CHUNKS
from backend.pickle_utils import load_from_pickle, store_to_pickle

logger = logging.getLogger(__name__)


def create_or_load_chunks(file_path: Path) -> List[Document]:
    """Ingest the resume PDF into Documents

    Args:
        file_path: Path to resume pdf file

    Returns:
        List of Documents
    """
    chunks_out_path = (PATH_CHUNKS / file_path.name).with_suffix(".pkl")
    if not chunks_out_path.exists():
        logger.info("%s: creating chunks", file_path)
        chunks = parse_from_unstructured(file_path)
        if not chunks_out_path.exists():
            chunks_out_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("%s: Storing chunks to this location", chunks_out_path)
        chunks_out_path.parent.mkdir(parents=True, exist_ok=True)
        store_to_pickle(chunks, chunks_out_path)
    else:
        logger.info("%s: Loading chunks from this location", chunks_out_path)
        chunks = load_from_pickle(chunks_out_path)

    return chunks


def parse_from_langchain_integration(file_path: Path) -> List[Document]:
    """Parse a pdf using LangChain's integration for Unstructered

    Args:
        file_path: Path to resume pdf file

    Returns:
        List of Documents
    """
    from langchain_community.document_loaders import UnstructuredPDFLoader

    logger.info(
        "%s: partitioning and chunking pdf using LangChain's UnstructuredPDFLoader",
        file_path,
    )
    loader = UnstructuredPDFLoader(file_path, mode="elements", strategy="hi_res")
    chunks: List[Document] = loader.load()
    # Filter metadata to successfully process into DB
    chunks = filter_complex_metadata(chunks)

    return chunks


def parse_from_unstructured(file_path: Path) -> List[Document]:
    """Parse a pdf using Unstructered

    Using Unstructured, partition a pdf, apply chunking, and return Document instances

    Args:
        file_path: Path to resume pdf file

    Returns:
        List of Documents
    """
    from unstructured.chunking.basic import chunk_elements
    from unstructured.partition.pdf import partition_pdf

    logger.info("%s: partitioning and chunking pdf using unstructured", file_path)
    elements = partition_pdf(filename=file_path, strategy="hi_res")
    chunked_elements = chunk_elements(elements)
    chunks = []
    for el in chunked_elements:
        chunks.append(
            Document(
                el.text,
                metadata={
                    "source": f"{el.metadata.file_directory}/{el.metadata.filename}",
                    "last_modified": el.metadata.last_modified,
                    "filetype": el.metadata.filetype,
                    "page_number": el.metadata.page_number,
                    "file_directory": el.metadata.file_directory,
                    "filename": el.metadata.filename,
                    "category": el.category,
                    "element_id": el.id,
                },
            )
        )

    return chunks
