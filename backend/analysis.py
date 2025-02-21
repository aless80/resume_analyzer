import logging
from typing import Dict, TypedDict

from langchain_core.prompts import PromptTemplate
from langchain_openai.chat_models.base import BaseChatOpenAI
from langgraph.constants import END, START
from langgraph.graph import StateGraph

from backend.configuration import Configuration

logger = logging.getLogger(__name__)


class State(TypedDict):
    full_resume: str
    job_description: str
    llm: BaseChatOpenAI
    skills_analysis: str
    grammar_analysis: str
    combined_output: str


def analyze_resume(
        full_resume: str, job_description: str, config: Configuration
) -> str:
    logger.info(
        "Start grammatical and skill analysis of resume using parallel workflow"
    )
    # Build workflow
    parallel_builder = StateGraph(State)
    # Add nodes
    parallel_builder.add_node("call_llm_skills_analysis", call_llm_skills_analysis)
    parallel_builder.add_node(
        "call_llm_grammatical_analysis", call_llm_grammatical_analysis
    )
    parallel_builder.add_node("aggregator", aggregator)

    # Add edges to connect nodes
    # Start ─┬─ call_llm_grammatical_analysis ───────────────────────┬── aggregator ── End
    #        └─ call_llm_skills_analysis ─────── aggregator_skills ──┘
    parallel_builder.add_edge(START, "call_llm_grammatical_analysis")
    parallel_builder.add_edge(START, "call_llm_skills_analysis")
    parallel_builder.add_edge("call_llm_skills_analysis", "aggregator")
    parallel_builder.add_edge("call_llm_grammatical_analysis", "aggregator")
    parallel_builder.add_edge("aggregator", END)
    parallel_workflow = parallel_builder.compile()

    # Invoke
    state: State = parallel_workflow.invoke(
        {
            "full_resume": full_resume,
            "job_description": job_description,
            "llm": config.llm,
        }
    )

    return state["combined_output"]


def call_llm_skills_analysis(state: State):
    # Template for analyzing the resume against the job description
    template_analysis = """
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
    2. Comma-delimited list of skills from the job description that match the resume
    3. Comma-delimited list of skills from the job description that are missing in the resume
    
    **Additional Comments**:
    Hard skills: comment on most important hard skills from the job description that are missing in the resume
    Soft skills: comment on most important soft skills from the job description that are missing in the resume
    Other tips
    Notify if the languages of resume and job description are not the same.
    
    Resume: ```{resume}```
    Job Description: ```{job_description}```

    Analysis:
    """
    prompt_analysis = PromptTemplate(
        input_variables=["resume", "job_description"], template=template_analysis
    )
    chain_analysis = prompt_analysis | state["llm"]
    response_analysis = chain_analysis.invoke(
        {"resume": state["full_resume"], "job_description": state["job_description"]}
    )

    return {"skills_analysis": response_analysis.content}


def call_llm_grammatical_analysis(state: State):
    template_grammar = """
    You are an AI assistant specialized in English and Norwegian languages. 
    Detect the language in the given resume, then detect any grammatical error.
    
    Example Response Structure:
    
    The resume is written in English/Norwegian and I could not detect significant grammatical errors.

    Resume: ```{resume}```
    """
    prompt_grammar = PromptTemplate(
        input_variables=["resume"], template=template_grammar
    )
    chain_grammar = prompt_grammar | state["llm"]
    response_grammar = chain_grammar.invoke({"resume": state["full_resume"]})

    return {"grammar_analysis": response_grammar.content}


def aggregator(state: State) -> Dict[str, str]:
    """Combine the joke and story into a single output"""
    combined = state["skills_analysis"]
    combined += "\n\n**Grammar**:\n" f"{state['grammar_analysis']}\n"

    return {"combined_output": combined}
