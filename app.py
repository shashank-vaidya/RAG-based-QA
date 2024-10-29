import streamlit as st
from services import extract_text_from_pdf, split_text_into_sentences, retrieve_top_relevant_chunks, generate_answer_with_gpt
from utils import load_openai_client, load_embedding_model

st.title("RAG-based QA with PDF or txt")

pdf_file = st.file_uploader("Upload your PDF", type="pdf")
question = st.text_input("Enter your question")

if pdf_file and question:
    client = load_openai_client()
    embedding_model = load_embedding_model()
    
    text = extract_text_from_pdf(pdf_file)
    document_chunks = split_text_into_sentences(text)
    top_chunks = retrieve_top_relevant_chunks(embedding_model, question, document_chunks)
    
    answer = generate_answer_with_gpt(client, question, top_chunks)
    st.subheader("Answer:")
    st.write(answer)
    
    st.subheader("Relevant Document Sections:")
    for chunk in top_chunks:
        st.write(chunk)
else:
    st.write("Please upload a PDF file and enter a question.")
