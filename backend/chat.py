from typing import Any, Dict, Literal, TypedDict

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.vectorstores import VectorStore
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langsmith import traceable
from pydantic import BaseModel, Field

from backend.configuration import Configuration, config_cache

config = Configuration()


def create_runnable_resume_chain(
    vector_store: VectorStore,
    job_description: str,
    config: Configuration,
    similarity_top_k: int = 4,
) -> RunnableWithMessageHistory:
    """Create a runnable chain for chatting with the resume and job description

    Args:
        vector_store: Vector store
        job_description: Job description
        config: Configuration object (optional)

    Returns:
        Runnable chain
    """
    retriever = vector_store.as_retriever(
        search_type="mmr",  # Maximum Marginal Relevance for search
        search_kwargs={
            "k": similarity_top_k,
            "fetch_k": min(20, vector_store._collection.count()),
        },
    )

    # Chat logic setup for contextualizing user questions
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    # Creating a prompt template for contextualizing questions
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # Creating a history-aware retriever with the language model
    history_aware_retriever = create_history_aware_retriever(
        config.llm, retriever, contextualize_q_prompt
    )

    # System prompt for answering questions
    system_prompt = (
        "You are an assistant specialized in resume analysis and recruitment. "
        "You are given pieces of retrieved context from a resume. "
        "Answer the question, if you don't know the answer, say that you don't know. "
        "Use three sentences maximum and keep the answer concise."
        "\n\n"
        "Context: ```{context}```"
    )

    # Creating a prompt template for question-answering
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("system", f"This is the job Description: {job_description}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # Setting up the question-answering chain
    question_answer_chain = create_stuff_documents_chain(config.llm, qa_prompt)
    retrieval_chain = create_retrieval_chain(
        retriever=history_aware_retriever, combine_docs_chain=question_answer_chain
    )

    # Chat history management using a dictionary
    store = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        # Create or return the chat history
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    # Creating a runnable chain with message history
    conversational_retrieval_chain = RunnableWithMessageHistory(
        retrieval_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return conversational_retrieval_chain


class State(TypedDict):
    """Graph state"""

    vector_store: VectorStore
    job_description: str
    messages: Dict[str, Any]
    query: str
    response: str
    attempt: int
    evaluation: str


class Feedback(BaseModel):
    """Class for structuring the evaluator's output"""

    evaluation: Literal["acceptable", "unacceptable"] = Field(
        description="Decide if the response is acceptable or not",
    )


evaluator = config.llm.with_structured_output(Feedback)


def llm_call_generator(state: State):
    """Initialize the chain to carry out a conversation, then generate a response"""
    similarity_top_k = config.similarity_top_k + (state["attempt"] - 1) * 2
    similarity_top_k = min(similarity_top_k, state["vector_store"]._collection.count())
    if state["attempt"] > 1:
        print(
            f"Assistant: I will try a new query with {similarity_top_k} retrieved documents"
        )

    conversational_retrieval_chain = create_runnable_resume_chain(
        vector_store=state["vector_store"],
        job_description=state["job_description"],
        config=config,
        similarity_top_k=similarity_top_k,
    )
    input_data = {
        "input": state["query"],
        "chat_history": state["messages"],
    }
    response_obj = conversational_retrieval_chain.invoke(
        input_data,
        config={"configurable": {"session_id": "TODO"}},
    )

    return {"response": response_obj["answer"], "attempt": state["attempt"] + 1}


def llm_call_evaluator(state: State):
    """LLM evaluates the output"""
    # Temporarily turn off cache due to deserialization issues with structured output
    config_cache(on=False)
    evaluation = evaluator.invoke(
        f"Evaluate the response `{state['response']}` "
        f"to the query `{state['query']}`"
    )
    config_cache(on=True)

    return {"evaluation": evaluation.evaluation}


def route_response(state: State):
    """Route back to response generator or end based upon the evaluator"""
    if state.get("evaluation", None) is None:
        raise ValueError("Response from evaluator is missing")
    if state["evaluation"] == "acceptable":
        return "Acceptable"
    elif state["evaluation"] == "unacceptable":
        if state["attempt"] > 3:
            return "Stop"
        return "Unacceptable"
    else:
        raise ValueError(
            f"{state['evaluation']}: Unexpected value in state['evaluation']"
        )


@traceable(
    run_type="llm",
    name="Resume Analyzer Decorator",
    project_name=config.langsmith_project,
)
def resume_chat_workflow(
    vector_store: VectorStore,
    job_description: str,
    query: str,
    messages: Dict[str, Any],
) -> str:
    """Chat using an evaluator-optimizer workflow

    Args:
        vector_store: Vector store for the resume
        job_description: Job description
        query: Query to the chat
        messages: Message history

    Returns:
        Response string
    """
    # Build workflow
    optimizer_builder = StateGraph(State)

    # Add the nodes
    optimizer_builder.add_node("llm_call_generator", llm_call_generator)
    optimizer_builder.add_node("llm_call_evaluator", llm_call_evaluator)

    # Add edges to connect nodes
    optimizer_builder.add_edge(START, "llm_call_generator")
    optimizer_builder.add_edge("llm_call_generator", "llm_call_evaluator")
    optimizer_builder.add_conditional_edges(
        "llm_call_evaluator",
        route_response,
        {
            "Acceptable": END,
            "Unacceptable": "llm_call_generator",
            "Stop": END,
        },
    )

    # Compile the workflow
    optimizer_workflow = optimizer_builder.compile()

    # Invoke
    state = optimizer_workflow.invoke(
        {
            "vector_store": vector_store,
            "job_description": job_description,
            "messages": messages,
            "query": query,
            "attempt": 1,
        }
    )

    return state["response"]  # type: ignore
