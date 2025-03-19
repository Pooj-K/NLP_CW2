# streamlit_app.py
import streamlit as st
import os
import numpy as np
import sentencepiece as spm
from gensim.models import FastText
from sklearn.neighbors import NearestNeighbors
import google.generativeai as genai
from dotenv import load_dotenv
import re

# Load environment variables from .env file
load_dotenv()

# Set up page configuration
st.set_page_config(page_title="Sri Lankan Constitution Chatbot", layout="wide")

# Title and description
st.title("Sri Lankan Constitution Chatbot")
st.markdown(
    "Ask questions about the Sri Lankan Constitution in Sinhala and get answers powered by AI with retrieval-augmented generation (RAG).")

# Load environment variables (e.g., API key)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error(
        "GOOGLE_API_KEY not found. Please set it in the .env file and ensure the file is in the project directory.")
    st.stop()
genai.configure(api_key=GOOGLE_API_KEY)

# Paths to models and data
FASTTEXT_MODEL_PATH = "fasttext_sinhala_bpe.model"
BPE_MODEL_PATH = "sinhala_spm_bpe.model"
CONSTITUTION_PATH = "Sri Lanka Constitution-Sinhala.txt"

# Load models and data with caching
@st.cache_resource
def load_models_and_data():
    ft_model = FastText.load(FASTTEXT_MODEL_PATH)
    sp = spm.SentencePieceProcessor(model_file=BPE_MODEL_PATH)
    with open(CONSTITUTION_PATH, "r", encoding="utf-8") as f:
        passages = [line.strip() for line in f if line.strip()]
    return ft_model, sp, passages


ft_model, sp, passages = load_models_and_data()


# Precompute passage embeddings and build nearest neighbor index
@st.cache_resource
def build_retrieval_index(passages, _ft_model, _sp):
    def get_embedding(text):
        tokens = _sp.encode(text, out_type=str)
        valid_vectors = [_ft_model.wv[t] for t in tokens if t in _ft_model.wv.key_to_index]
        return np.mean(valid_vectors, axis=0) if valid_vectors else np.zeros(_ft_model.vector_size)

    passage_embeddings = np.array([get_embedding(p) for p in passages])
    nn = NearestNeighbors(n_neighbors=10, metric="cosine")
    nn.fit(passage_embeddings)
    return nn, passage_embeddings


nn, passage_embeddings = build_retrieval_index(passages, ft_model, sp)

# Initialize Gemini model
model = genai.GenerativeModel("gemini-1.5-pro")


# Function to retrieve context
def retrieve_context(question, nn, passages, ft_model, sp):
    def get_embedding(text):
        tokens = sp.encode(text, out_type=str)
        valid_vectors = [ft_model.wv[t] for t in tokens if t in ft_model.wv.key_to_index]
        return np.mean(valid_vectors, axis=0) if valid_vectors else np.zeros(ft_model.vector_size)

    query_emb = get_embedding(question).reshape(1, -1)
    distances, indices = nn.kneighbors(query_emb)

    # Return both context and the cosine similarity scores (as distances)
    retrieved_passages = [passages[i] for i in indices[0]]
    similarity_scores = [1 - dist for dist in distances[0]]  # Convert distance to similarity (1 - distance)

    return retrieved_passages, similarity_scores


# Sri Lanka Constitution keywords in English and Sinhala
CONSTITUTION_KEYWORDS = [
    # English keywords
    r'constitution', r'law', r'parliament', r'president', r'judiciary', r'government', r'election', r'vote',
    r'court', r'rights', r'article', r'amendment', r'sri lanka', r'citizen', r'minister', r'cabinet',
    # Sinhala keywords
    r'ව්‍යවස්ථාව', r'පාර්ලිමේන්තු', r'ජනාධිපති', r'අධිකරණ', r'ආණ්ඩුව', r'මැතිවරණ', r'ඡන්ද',
    r'උසාවි', r'අයිතිවාසිකම්', r'වගන්ති', r'සංශෝධන', r'ශ්‍රී ලංකා', r'පුරවැසි', r'අමාත්‍ය', r'කැබිනට්',
    # Added additional language-related keywords
    r'භාෂාව', r'ජාතික භාෂා', r'රාජ්‍ය භාෂාව', r'language', r'national language', r'official language',
    r'සිංහල', r'දෙමළ', r'ඉංග්‍රීසි', r'sinhala', r'tamil', r'english'
]


# Function to generate answer with Gemini API
def generate_answer(prompt, max_tokens=700):
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=0.2  # Slightly increased for better completions
        )
    )
    return response.text


# Function to answer with RAG - removed out-of-scope detection
def answer_with_rag(question, nn, passages, ft_model, sp):
    retrieved_passages, similarity_scores = retrieve_context(question, nn, passages, ft_model, sp)
    context = " ".join(retrieved_passages)

    # Enhanced prompt for better results
    prompt = (
        f"Instructions: You are an expert on the Sri Lankan Constitution. Use the provided context to answer the question about the Sri Lankan Constitution accurately and completely in Sinhala. "
        f"Even if the context doesn't have explicit information about the question, if it contains related information, use it to provide the best possible answer. "
        f"For questions about national languages, official languages, or language rights, be especially thorough in your answer. "
        f"Include specific references to articles or sections of the Sri Lankan Constitution when possible. "
        f"If the context mentions languages (Sinhala, Tamil, English) in any capacity, include that information in your answer. "
        f"If there is absolutely no relevant information in the context, respond with: 'සමාවන්න, මට මෙම ප්‍රශ්නයට පිළිතුරු දීමට ප්‍රමාණවත් තොරතුරු නොමැත.'\n"
        f"Context: {context}\n"
        f"Q: {question}\n"
        f"A:"
    )

    return generate_answer(prompt, max_tokens=700), context, similarity_scores


# Function to check if a query needs pre-processing
def preprocess_query(query):
    # Map of common question patterns about national/official languages
    language_questions = {
        r'ජාතික භාෂා': 'ශ්‍රී ලංකා ව්‍යවස්ථාව අනුව ජාතික භාෂා මොනවාද? සිංහල, දෙමළ, ඉංග්‍රීසි භාෂා ගැන ව්‍යවස්ථාවේ කුමක් සඳහන් වේද?',
        r'රාජ්‍ය භාෂාව': 'ශ්‍රී ලංකා ව්‍යවස්ථාව අනුව රාජ්‍ය භාෂා මොනවාද? සිංහල, දෙමළ, ඉංග්‍රීසි භාෂා ගැන ව්‍යවස්ථාවේ කුමක් සඳහන් වේද?',
        r'නිල භාෂා': 'ශ්‍රී ලංකා ව්‍යවස්ථාව අනුව නිල භාෂා මොනවාද? සිංහල, දෙමළ, ඉංග්‍රීසි භාෂා ගැන ව්‍යවස්ථාවේ කුමක් සඳහන් වේද?',
        r'භාෂා': 'ශ්‍රී ලංකා ව්‍යවස්ථාව අනුව භාෂා පිළිබඳව කුමක් සඳහන් වේද? සිංහල, දෙමළ, ඉංග්‍රීසි භාෂා ගැන ව්‍යවස්ථාවේ කුමක් සඳහන් වේද?'
    }

    # Check if the query matches any patterns
    for pattern, expanded_query in language_questions.items():
        if re.search(pattern, query, re.IGNORECASE):
            return expanded_query

    return query  # Return original if no pattern matches


# Chat history management
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input form for user questions
with st.form(key="question_form", clear_on_submit=True):
    question = st.text_input("Enter your question in Sinhala:",
                             placeholder="E.g., ශ්‍රී ලංකා ව්‍යවස්ථාව අනුව ජාතික භාෂා මොනවාද?")
    submit_button = st.form_submit_button(label="Ask")

# Process the question and display the answer
if submit_button and question:
    # Preprocess the query if needed
    processed_question = preprocess_query(question)

    answer, context, similarity_scores = answer_with_rag(processed_question, nn, passages, ft_model, sp)

    # Store the original question, not the processed one in the chat history
    st.session_state.chat_history.append({
        "question": question,
        "answer": answer,
        "context": context,
        "relevance": max(similarity_scores)
    })

# Display chat history
if st.session_state.chat_history:
    st.subheader("Chat History")
    for i, chat in enumerate(st.session_state.chat_history):
        with st.container():
            st.markdown(f"**Q{i + 1}:** {chat['question']}")
            st.markdown(f"**A{i + 1}:** {chat['answer']}")
            with st.expander("View Retrieved Context"):
                st.markdown(f"{chat['context']}")
                st.markdown(f"Relevance score: {chat['relevance']:.4f}")
            st.markdown("---")