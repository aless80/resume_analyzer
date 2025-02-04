import streamlit as st

from backend.chat import chat


# Chat interface section of the application - displayed at the right
def render_chat_interface():
    st.header("Chat with the Resume")  # Header for the chat interface

    # Add CSS for fixing chat input position at the bottom
    st.markdown(
        """
        <style>
        .stChatInput {
            position: fixed;
            bottom: 0;   
            padding: 1rem;
            background-color: white;
            z-index: 1000;
        }
        .stChatFloatingInputContainer {
            margin-bottom: 20px;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )  # Injecting custom CSS for styling

    # Initialize empty chat messages
    if "messages" not in st.session_state:
        st.session_state.messages = []  # Initialize messages in session state

    # Check if the vector store is available
    if "vector_store" in st.session_state and "job_description" in st.session_state:
        # Setting up the vector store as retriever
        conversational_retrieval_chain = chat(
            st.session_state.vector_store, st.session_state.job_description
        )

        # Create a container for messages with bottom padding for input space
        chat_container = st.container()

        # Add space at the bottom to prevent messages from being hidden behind input
        st.markdown("<div style='height: 100px;'></div>", unsafe_allow_html=True)

        # Input box - will be fixed at bottom due to CSS
        prompt = st.chat_input("Ask about the resume")  # Input for user queries

        # Display messages in the container
        with chat_container:
            for (
                message
            ) in st.session_state.messages:  # Iterate through session messages
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])  # Display message content

        if prompt:  # Check if there is a user input
            st.session_state.messages.append(
                {"role": "user", "content": prompt}
            )  # Store user message
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)  # Display user input

                with st.chat_message("assistant"):
                    # Prepare input data for the conversational chain
                    input_data = {
                        "input": prompt,
                        "chat_history": st.session_state.messages,
                    }
                    response = conversational_retrieval_chain.invoke(
                        input_data,
                        config={
                            "configurable": {
                                "session_id": "abc123"
                            }  # Setting session ID
                        },
                    )
                    answer_text = response["answer"]
                    st.markdown(answer_text)

            st.session_state.messages.append(
                {"role": "assistant", "content": answer_text}
            )  # Store assistant response

            # Force a rerun to update the chat immediately
            st.rerun()  # Refresh the Streamlit app

    else:
        st.info("Please upload a resume and analyze it to start chatting.")
