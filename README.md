# Resume Analyzer
This is a RAG-based resume analysis tool built using Streamlit for frontend and LangChain for backend. 
It includes two main tasks: analysis of a resume based on a job description, and 
implementation of a conversational retrieval chain to chat with the resume and job description.

The analysis task checks for grammatical issues, compares and evaluates the skills present in the resume 
with the job description, and suggestions to the applicant.   

The chat functionality is and LLM augmented with retrieval capabilities on the resume and 
memory about the job description.

Install two Linux packages, then use pip:
```
apt-get install poppler-utils tesseract-ocr
pip install -r requirements.txt
```
Create and setup a `.env` file from a template:
```
cp .env_mock .env
```

Run with python or streamlit:
```
python main.py
streamlit run app.py
```
