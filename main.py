import logging
from typing import Any, Dict

from backend import configure_loggers_levels
from backend.analysis import analyze_resume
from backend.chat import resume_chat_workflow
from backend.configuration import Configuration, config_cache
from backend.logging_config import LOGGER_CONFIG
from backend.pdf_ingestion import create_or_load_chunks
from backend.vector_store import create_or_load_vector_store

logging.config.dictConfig(LOGGER_CONFIG)


def main():

    config = Configuration()
    configure_loggers_levels(config.logger_level)
    config_cache()

    resume_file_path = config.cv_path
    job_description = config.job_description

    # Create a temporary directory for the cv file
    temp_dir = Path("temp")
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Load and split the PDF file into documents and chunks
    chunks = create_or_load_chunks(resume_file_path)
    # Combine all document contents into one text string for analysis
    full_resume = " ".join([doc.page_content for doc in chunks])

    # Analyze the resume
    response_analysis = analyze_resume(full_resume, job_description, config=config)
    print(response_analysis)

    ### Chat
    # Create a vector store from the resume chunks
    vector_store = create_or_load_vector_store(
        chunks=chunks,
        vector_index_name=resume_file_path.name.removesuffix(".pdf"),
        config=config,
    )

    print(
        f"{Color.BOLD}\nWelcome to CV analyzer. Type your query or type 'exit' to quit{Color.END}"
    )
    messages: Dict[str, Any] = []
    while True:
        query = input(f"{Color.BOLD}User: {Color.END}")

        if query == "":
            continue
        elif query.lower() == "exit":
            break
        else:
            # Prepare input data for the conversational chain
            answer_text = resume_chat_workflow(
                vector_store=vector_store,
                job_description=job_description,
                query=query,
                messages=messages,
            )

            print(
                f"{Color.BOLD}Assistant: {answer_text}{Color.END}",
            )
            messages.append({"role": "user", "content": query})
            messages.append({"role": "assistant", "content": answer_text})


class Color:
    BLUE = "\033[94m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    ITALIC = "\033[3m"
    END = "\033[0m"


# Script execution through the 'main' function
if __name__ == "__main__":
    main()
