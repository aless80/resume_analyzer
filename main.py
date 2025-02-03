import os
from pathlib import Path

import dotenv

from backend.analysis import analyze_resume
from backend.chat import chat
from backend.configuration import Configuration, config_cache, config_tracing
from backend.pdf_ingestion import create_or_load_chunks
from backend.vector_store import create_or_load_vector_store

dotenv.load_dotenv()

RESUME_FILE_PATH = Path(os.getenv("CV_PATH"))
JOB_DESCRIPTION = os.getenv("JOB_DESCRIPTION")


def main(
    resume_file_path: Path = RESUME_FILE_PATH, job_description: str = JOB_DESCRIPTION
):
    config = Configuration()
    config_tracing(config)
    config_cache()

    # Create a temporary directory for the cv file
    temp_dir = Path("temp")
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Save the uploaded file to the temporary directory
    # with open(temp_dir / resume_file_path, "wb") as f:
    #     f.write(resume_file_path.getbuffer())

    # Load and split the PDF file into documents and chunks
    chunks = create_or_load_chunks(resume_file_path, config=config)

    # Remove the temporary directory and its contents
    # shutil.rmtree(temp_dir)

    # Button to begin resume analysis
    # Combine all document contents into one text string for analysis
    full_resume = " ".join([doc.page_content for doc in chunks])

    # Analyze the resume
    response_analysis = analyze_resume(full_resume, job_description, config=config)
    print(response_analysis.content)

    ### Chat
    # Create a vector store from the resume chunks
    vector_store = create_or_load_vector_store(
        chunks=chunks, vector_index_name=resume_file_path.name.removesuffix(".pdf")
    )
    # Initialize the chain to carry out a conversation
    conversational_retrieval_chain = chat(vector_store=vector_store)

    print(
        f"{Color.BOLD}\nWelcome to CV analyzer. Type your query or type 'exit' to quit{Color.END}"
    )
    messages = []
    while True:
        query = input(f"{Color.BOLD}User: {Color.END}")

        if query == "":
            continue
        elif query.lower() == "exit":
            break
        else:
            # Prepare input data for the conversational chain
            input_data = {
                "input": query,
                "chat_history": messages,
            }
            response_obj = conversational_retrieval_chain.invoke(
                input_data,
                config={"configurable": {"session_id": "abc123"}},  # Setting session ID
            )

            answer_text = response_obj["answer"]
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
