from typing import Any

from langchain_core.prompts import PromptTemplate

from backend.configuration import Configuration


def analyze_resume(full_resume, job_description, config: Configuration) -> Any:
    # Template for analyzing the resume against the job description
    template = """
    You are an AI assistant specialized in resume analysis and recruitment. 
    Analyze the given resume and compare it with the job description. 
    
    Example Response Structure:
    
    **OVERVIEW**:
    - **Match Percentage**: [Calculate overall match percentage between the resume and job description]
    - **Matched Skills**: [List the skills in job description that match the resume]
    - **Unmatched Skills**: [List the skills in the job description that are missing in the resume]

    **DETAILED ANALYSIS**:
    Provide a detailed analysis about:
    1. Overall match percentage between the resume and job description
    2. List of skills from the job description that match the resume
    3. List of skills from the job description that are missing in the resume
    
    **Additional Comments**:
    Additional comments about the resume and suggestions for the applicant.
    
    **Grammar**:
    Detect the language and show any grammatical error.

    Resume: ```{resume}```
    Job Description: ```{job_description}```

    Analysis:
    """
    prompt = PromptTemplate(
        input_variables=["resume", "job_description"], template=template
    )

    # Create a chain combining the prompt and the language model
    chain = prompt | config.llm

    # Invoke the chain with input data
    response = chain.invoke({"resume": full_resume, "job_description": job_description})

    return response
