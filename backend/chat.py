from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory

from backend.configuration import Configuration


def chat(vector_store, job_description: str = "") -> RunnableWithMessageHistory:
    config = Configuration()

    retriever = vector_store.as_retriever(
        search_type="mmr",  # Uses Maximum Marginal Relevance for search
        search_kwargs={"k": config.similarity_top_k},
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
        history_aware_retriever, question_answer_chain
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
