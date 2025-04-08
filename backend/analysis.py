import logging
from typing import Dict, TypedDict

from langchain_core.prompts import PromptTemplate
from langchain_openai.chat_models.base import BaseChatOpenAI
from langgraph.constants import END, START
from langgraph.graph import StateGraph

from backend.configuration import Configuration

logger = logging.getLogger(__name__)


class State(TypedDict):
    """State dictionary for the parallel workflow"""

    full_resume: str
    job_description: str
    llm: BaseChatOpenAI
    skills_analysis: str
    grammar_analysis: str
    style_analysis: str
    combined_output: str


def analyze_resume(
    full_resume: str, job_description: str, config: Configuration
) -> str:
    """Analyze the resume against the job description using parallel workflow

    Args:
        full_resume: Resume text
        job_description: Job description text
        config: Configuration object

    Returns:
        Combined analysis of the resume
    """
    logger.info(
        "Start grammatical and skill analysis of resume using parallel workflow"
    )
    # Build workflow
    parallel_builder = StateGraph(State)
    # Add nodes
    parallel_builder.add_node("call_llm_skills_analysis", call_llm_skills_analysis)
    parallel_builder.add_node("call_llm_style_analysis", call_llm_style_analysis)
    parallel_builder.add_node(
        "call_llm_grammatical_analysis", call_llm_grammatical_analysis
    )
    parallel_builder.add_node("aggregator", aggregator)

    # Add edges to connect nodes
    # Start ─┬──── call_llm_grammatical_analysis ───────────────────────┬── aggregator ── End
    #        ├──── call_llm_style_analysis  ────────────────────────────┤
    #        └ ─ ─ call_llm_skills_analysis ────────────────────────────┘
    parallel_builder.add_edge(START, "call_llm_grammatical_analysis")
    parallel_builder.add_edge(START, "call_llm_style_analysis")
    parallel_builder.add_conditional_edges(
        START, check_job_description, {True: "call_llm_skills_analysis", False: END}
    )
    parallel_builder.add_edge("call_llm_skills_analysis", "aggregator")
    parallel_builder.add_edge("call_llm_style_analysis", "aggregator")
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


def check_job_description(state: State) -> bool:
    """Gate function to check if the job description is present

    Args:
        state: State dictionary for the parallel workflow

    Returns:
        Boolean indicating if the job description is present
    """
    if state["job_description"] == "":
        return False
    return True


def call_llm_skills_analysis(state: State) -> Dict[str, str]:
    """Analyze the skills in the resume against the job description

    Args:
        state: State dictionary for the parallel workflow

    Returns:
        Skills analysis of the resume
    """
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
    Notify if the languages of resume and job description are not the same, otherwise skip this part. 
    
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


def call_llm_grammatical_analysis(state: State) -> Dict[str, str]:
    """Analyze the grammar of the resume

    Args:
        state: State dictionary for the parallel workflow

    Returns:
        Grammar analysis of the resume
    """
    template_grammar = """
    You are an AI assistant specialized in English and Norwegian languages. 
    Detect the language in the given resume, then detect any grammatical error.
    
    Example Response Structure:
    * "optimzing" should be "optimizing"
    * "Alessandro do a great job" should be "Alessandro does a great job"
    OR
    The resume is written in English/Norwegian and I could not detect significant grammatical errors.

    Resume: ```{resume}```
    """
    prompt_grammar = PromptTemplate(
        input_variables=["resume"], template=template_grammar
    )
    chain_grammar = prompt_grammar | state["llm"]
    response_grammar = chain_grammar.invoke({"resume": state["full_resume"]})

    return {"grammar_analysis": response_grammar.content}


def call_llm_style_analysis(state: State) -> Dict[str, str]:
    """Analyze the grammar of the resume

    Args:
        state: State dictionary for the parallel workflow

    Returns:
        Stylistic analysis of the resume
    """
    template_style = """
    You are an AI assistant specialized in English and Norwegian languages. 
    Detect the language in the given resume, then suggest any stylistic improvements.
    Focus on:    
    * Action Verbs: Begin resume section with strong action verbs like: managed, created, developed, 
      improved, or enhanced.
    * Active Formulation: Instead of saying "responsible for," say "led," "oversaw," or "directed."
    * Avoid Jargon: Use clear, understandable language to ensure your message is conveyed 
      effectively to all readers.
    * Cut Redundancy: Remove unnecessary words and phrases to keep your CV concise and impactful.
    
    Do not comment on formatting, spelling, or punctuation.

    Example Response Structure:
    * Use action verbs: "Alessandro managed" instead of "was responsible for managing".
    * Use active formulation: "led a project" instead "was responsible for development of a project"
    * Cut redundancy: expertise in Python is repeated two times in the same section. 

    Resume: ```{resume}```
    """
    prompt_style = PromptTemplate(input_variables=["resume"], template=template_style)
    chain_grammar = prompt_style | state["llm"]
    response_style = chain_grammar.invoke({"resume": state["full_resume"]})

    return {"style_analysis": response_style.content}


def aggregator(state: State) -> Dict[str, str]:
    """Combine the analysis parts into a single output

    Args:
        state: State dictionary for the parallel workflow

    Returns:
        Combined output of the analysis
    """
    combined = ""
    if "skills_analysis" in state:
        combined += state["skills_analysis"]
    combined += "\n\n**Grammar**:\n" f"{state['grammar_analysis']}\n"
    combined += "\n\n**Style**:\n" f"{state['style_analysis']}\n"

    return {"combined_output": combined}
