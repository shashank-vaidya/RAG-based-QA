import re
import pdfplumber
from sentence_transformers import util

def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def split_text_into_sentences(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]

def retrieve_top_relevant_chunks(embedding_model, question, document_chunks, top_k_initial=10, top_k_final=5):
    question_embedding = embedding_model.encode(question, convert_to_tensor=True)
    chunk_embeddings = embedding_model.encode(document_chunks, convert_to_tensor=True)
    similarities = util.cos_sim(question_embedding, chunk_embeddings)[0]
    
    top_k_initial = min(top_k_initial, len(document_chunks))
    top_k_indices = similarities.topk(top_k_initial).indices
    initial_top_chunks = [document_chunks[idx] for idx in top_k_indices]
    
    refined_chunk_embeddings = embedding_model.encode(initial_top_chunks, convert_to_tensor=True)
    refined_similarities = util.cos_sim(question_embedding, refined_chunk_embeddings)[0]
    
    top_k_final = min(top_k_final, len(initial_top_chunks))
    top_k_final_indices = refined_similarities.topk(top_k_final).indices
    final_top_chunks = [initial_top_chunks[idx] for idx in top_k_final_indices]
    
    return final_top_chunks

def generate_answer_with_gpt(client, question, top_chunks, model="gpt-4o"):
    context = "\n\n".join(top_chunks)
    prompt = (
        f"Context: {context}\n\n"
        f"Question: {question}\n\n"
        "Please answer the question based on the provided context."
    )
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant for answering questions based on document context."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()
