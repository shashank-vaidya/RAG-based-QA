import re
import pdfplumber
from sentence_transformers import SentenceTransformer, util
import chromadb
import os
import nltk
from nltk.corpus import stopwords, wordnet
import numpy as np

nltk.data.path.append(r"C:/Users/s.vaidya/AppData/Roaming/nltk_data")
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))

chroma_db_path = os.path.join(os.getcwd(), "chroma_db")
chroma_client = chromadb.PersistentClient(path=chroma_db_path)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def split_text_into_paragraphs(text, chunk_size=120, overlap=30):
    original_paragraphs = []
    cleaned_paragraphs = []
    
    words = nltk.word_tokenize(text.lower())
    cleaned_words = [word for word in words if word.isalnum() and word not in stop_words]
    original_text_words = text.split()

    for i in range(0, len(original_text_words), chunk_size - overlap):
        original_paragraph = " ".join(original_text_words[i:i + chunk_size])
        original_paragraphs.append(original_paragraph)

        cleaned_paragraph = " ".join(cleaned_words[i:i + chunk_size])
        cleaned_paragraphs.append(cleaned_paragraph)

    return {"cleaned": cleaned_paragraphs, "original": original_paragraphs}

def store_paragraphs_in_chromadb(paragraphs):
    try:
        chroma_client.delete_collection("document_paragraphs")
    except ValueError:
        print("Collection 'document_paragraphs' does not exist; creating it instead.")
    
    global chroma_collection
    chroma_collection = chroma_client.get_or_create_collection("document_paragraphs")

    for idx, paragraph in enumerate(paragraphs):
        embedding = embedding_model.encode(paragraph)
        chroma_collection.add(
            documents=[paragraph],
            embeddings=[embedding],
            metadatas=[{"paragraph_id": idx}],
            ids=[str(idx)]
        )

def extract_keywords(question):
    words = nltk.word_tokenize(question)
    pos_tags = nltk.pos_tag(words)
    keywords = set()

    for word, tag in pos_tags:
        if tag.startswith('NN') or tag.startswith('VB'):
            keywords.add(word.lower())
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    keywords.add(lemma.name().lower())

    return keywords

def filter_paragraphs_by_keywords(paragraphs, keywords):
    return [paragraph for paragraph in paragraphs if any(keyword in paragraph.lower() for keyword in keywords)]

def generate_question_variations(client, question, model="gpt-4o", num_variations=3):
    prompt = f"Generate {num_variations} variations of the following question:\nQuestion: {question}"
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    
    variations_text = response.choices[0].message.content.strip().split("\n")
    return [variation.strip("- ").strip() for variation in variations_text[:num_variations]]

def retrieve_relevant_paragraphs(client, question, embedding_model, paragraph_dict, n_paragraphs=5):
    paragraphs = paragraph_dict["cleaned"]
    original_paragraphs = paragraph_dict["original"]
    
    keywords = extract_keywords(question)
    relevant_cleaned_paragraphs = filter_paragraphs_by_keywords(paragraphs, keywords)
    if not relevant_cleaned_paragraphs:
        relevant_cleaned_paragraphs = paragraphs

    question_variations = [question]
    question_variations += generate_question_variations(client, question)

    all_similarity_scores = []
    for variation in question_variations:
        variation_embedding = embedding_model.encode(variation)
        for paragraph in relevant_cleaned_paragraphs:
            paragraph_embedding = embedding_model.encode(paragraph)
            all_similarity_scores.append(util.cos_sim(variation_embedding, paragraph_embedding).item())

    dynamic_threshold = np.percentile(all_similarity_scores, 75) if all_similarity_scores else 0.1
    print(f"Dynamic similarity threshold set at: {dynamic_threshold:.4f}")

    scored_paragraphs = []
    for variation in question_variations:
        variation_embedding = embedding_model.encode(variation)
        for i, paragraph in enumerate(relevant_cleaned_paragraphs):
            paragraph_embedding = embedding_model.encode(paragraph)
            similarity_score = util.cos_sim(variation_embedding, paragraph_embedding).item()
            if similarity_score >= dynamic_threshold:
                keyword_matches = sum(1 for word in keywords if word in paragraph.lower())
                score = similarity_score + 0.05 * keyword_matches
                scored_paragraphs.append((original_paragraphs[i], score))

    scored_paragraphs = sorted(scored_paragraphs, key=lambda x: x[1], reverse=True)
    return [paragraph for paragraph, _ in scored_paragraphs[:n_paragraphs]]

def similarity_check(question, paragraphs):
    question_embedding = embedding_model.encode(question)
    print("Similarity scores between question and each chunk:")
    for idx, paragraph in enumerate(paragraphs):
        paragraph_embedding = embedding_model.encode(paragraph)
        similarity_score = util.cos_sim(question_embedding, paragraph_embedding).item()
        print(f"Chunk {idx} similarity: {similarity_score:.4f}")

def generate_answer_with_gpt(client, question, paragraphs, model="gpt-4o"):
    context = "\n\n".join(paragraphs)
    prompt = (
        f"Context: {context}\n\n"
        f"Question: {question}\n\n"
        "Please provide an answer based on the context provided."
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()
