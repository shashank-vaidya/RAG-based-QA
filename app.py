import streamlit as st
from services import (
    extract_text_from_pdf,
    split_text_into_paragraphs,
    store_paragraphs_in_chromadb,
    retrieve_relevant_paragraphs,
    generate_answer_with_gpt,
    similarity_check
)
from utils import load_openai_client, load_embedding_model

st.title("RAG-based QA with Paragraph-Level Retrieval")

pdf_file = st.file_uploader("Upload your PDF", type="pdf")
question = st.text_input("Enter your question")

if pdf_file and question:
    client = load_openai_client()
    embedding_model = load_embedding_model()
    
    text = extract_text_from_pdf(pdf_file)
    paragraph_dict = split_text_into_paragraphs(text)
    
    store_paragraphs_in_chromadb(paragraph_dict["cleaned"])
    
    similarity_check(question, paragraph_dict["cleaned"])
    
    top_original_paragraphs = retrieve_relevant_paragraphs(client, question, embedding_model, paragraph_dict, n_paragraphs=5)
    
    answer = generate_answer_with_gpt(client, question, top_original_paragraphs)
    
    st.subheader("Answer:")
    st.write(answer)
    
    st.subheader("Relevant Document Sections:")
    if top_original_paragraphs:
        for idx, paragraph in enumerate(top_original_paragraphs, start=1):
            st.markdown(f"**Section {idx}:**")
            st.write(paragraph)
            st.markdown("---")  
    else:
        st.write("No relevant sections found.")
else:
    st.write("Please upload a PDF file and enter a question.")
