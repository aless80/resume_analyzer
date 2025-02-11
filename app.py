import streamlit as st

from backend.configuration import Configuration
from frontend.chat_interface import render_chat_interface
from frontend.main_app import render_main_app

# Set the page layout to wide for better visual presentation
st.set_page_config(layout="wide")


def main():
    config = Configuration()

    st.title("Resume Analyzer")

    with st.sidebar:
        st.image("resume_analyzer_logo.png", width=150)

    # Create two columns with a 3:2 ratio for layout
    col1, col2 = st.columns([3, 2])

    with col1:
        # Render the main app in the larger column
        render_main_app(config=config)

    with col2:
        # Render the chat interface in the smaller column
        render_chat_interface(config=config)


# Script execution through the 'main' function
if __name__ == "__main__":
    main()
