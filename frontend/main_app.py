import shutil
from pathlib import Path

import streamlit as st

from backend.analysis import analyze_resume
from backend.configuration import Configuration
from backend.pdf_ingestion import create_or_load_chunks
from backend.vector_store import create_or_load_vector_store

config = Configuration()


# Main application including "Upload Resume" and "Resume Analysis" sections
def render_main_app():
    # Apply custom CSS to adjust the sidebar width
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            min-width: 25%;
            max-width: 25%;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Moving the upload section to the sidebar
    with st.sidebar:
        st.header("Upload Resume")  # Header for the upload section

        # File uploader for PDF resumes
        resume_file = st.file_uploader("Upload Resume (PDF)", type="pdf")
        # Text area for job description input
        job_description = st.text_area(
            "Enter Job Description", height=300, placeholder="default"
        )

        if resume_file and job_description:  # Check if both inputs are provided
            # Create a temporary directory if it doesn't exist
            temp_dir = Path("temp")
            temp_dir.mkdir(parents=True, exist_ok=True)

            # Save the uploaded file to the temporary directory
            with open(temp_dir / resume_file.name, "wb") as f:
                f.write(resume_file.getbuffer())

            # Load and split the PDF file into documents and chunks
            resume_file_path = Path("temp") / resume_file.name
            chunks = create_or_load_chunks(resume_file_path, config=config)

            # Create a vector store from the resume chunks
            vector_store = create_or_load_vector_store(
                chunks=chunks, vector_index_name=resume_file.name
            )
            st.session_state.vector_store = (
                vector_store  # Store vector store in session state
            )

            # Remove the temporary directory and its contents
            shutil.rmtree(temp_dir)

            # Button to begin resume analysis
            if st.button("Analyze Resume", help="Click to analyze the resume"):
                # Combine all document contents into one text string for analysis
                full_resume = " ".join([doc.page_content for doc in chunks])
                # Analyze the resume
                analysis = analyze_resume(full_resume, job_description, config=config)
                # Store analysis in session state
                st.session_state.analysis = analysis
        else:
            st.info("Please upload a resume and enter a job description to begin.")

    # Display the analysis result if it exists in session state
    if "analysis" in st.session_state:
        st.header("Resume-Job Compatibility Analysis")
        st.write(st.session_state.analysis)
    else:
        st.header("Welcome to the Ultimate Resume Analysis Tool!")
        st.subheader("Your one-stop solution for resume screening and analysis.")
        st.info(
            "Do you want to find out the compatibility between a resume and a job description? So what are you waiting for?"
        )

        todo = ["Upload a Resume", "Enter a Job Description", "Click on Analyze Resume"]
        st.markdown(
            "\n".join([f"##### {i + 1}. {item}" for i, item in enumerate(todo)])
        )
