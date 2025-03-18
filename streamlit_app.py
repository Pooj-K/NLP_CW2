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
FASTTEXT_MODEL_PATH = "C:/Users/K.Hewageegana/OneDrive - CGIAR/Documents/Kanishka/chatbot_llm/fasttext_sinhala_bpe.model"
BPE_MODEL_PATH = "C:/Users/K.Hewageegana/OneDrive - CGIAR/Documents/Kanishka/chatbot_llm/sinhala_spm_bpe.model"
CONSTITUTION_PATH = "C:/Users/K.Hewageegana/OneDrive - CGIAR/Documents/Kanishka/chatbot_llm/Sri Lanka Constitution-Sinhala.txt"


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
def build_retrieval_index(passages, _ft_model, _sp):  # Added underscores to both parameters
    def get_embedding(text):
        tokens = _sp.encode(text, out_type=str)
        valid_vectors = [_ft_model.wv[t] for t in tokens if t in _ft_model.wv.key_to_index]
        return np.mean(valid_vectors, axis=0) if valid_vectors else np.zeros(_ft_model.vector_size)

    passage_embeddings = np.array([get_embedding(p) for p in passages])
    nn = NearestNeighbors(n_neighbors=3, metric="cosine")
    nn.fit(passage_embeddings)
    return nn, passage_embeddings


nn, passage_embeddings = build_retrieval_index(passages, ft_model, sp)

# Initialize Gemini model
model = genai.GenerativeModel("gemini-1.5-flash")


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
    r'උසාවි', r'අයිතිවාසිකම්', r'වගන්ති', r'සංශෝධන', r'ශ්‍රී ලංකා', r'පුරවැසි', r'අමාත්‍ය', r'කැබිනට්'
]

# Compile the keywords into a regex pattern
KEYWORD_PATTERN = re.compile('|'.join(CONSTITUTION_KEYWORDS), re.IGNORECASE)


# Function to check if a question is relevant to the Sri Lankan Constitution
def is_relevant_question(question, retrieved_passages, similarity_scores):
    # Check if the question contains any constitution-related keywords
    keyword_match = bool(KEYWORD_PATTERN.search(question))

    # Check if any of the retrieved passages are meaningful (not just placeholder or empty text)
    meaningful_context = False
    for passage in retrieved_passages:
        # Check if passage contains actual content, not just numbers or equals signs
        if len(passage.strip()) > 20 and not passage.strip().startswith('===') and not re.match(r'^\([iv]+\)',
                                                                                                passage.strip()):
            meaningful_context = True
            break

    # Check if the highest similarity score is above threshold
    similarity_threshold = 0.75  # Adjusted higher
    high_similarity = max(similarity_scores) >= similarity_threshold

    # Consider it relevant if it has keywords AND either meaningful context or high similarity
    return keyword_match and (meaningful_context or high_similarity)


# Function to generate answer with Gemini API
def generate_answer(prompt, max_tokens=300):
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=0.0
        )
    )
    return response.text


# Out of scope message in Sinhala
OUT_OF_SCOPE_MESSAGE = "සමාවන්න, ඔබේ ප්‍රශ්නය ශ්‍රී ලංකා ආණ්ඩුක්‍රම ව්‍යවස්ථාවට අදාළ නොවේ. මෙම චැට්බොට් ශ්‍රී ලංකා ආණ්ඩුක්‍රම ව්‍යවස්ථාව පිළිබඳ ප්‍රශ්න සඳහා පමණක් පිළිතුරු සපයයි. කරුණාකර ශ්‍රී ලංකා ආණ්ඩුක්‍රම ව්‍යවස්ථාවට අදාළ ප්‍රශ්නයක් අසන්න."


# Function to answer with RAG
def answer_with_rag(question, nn, passages, ft_model, sp):
    retrieved_passages, similarity_scores = retrieve_context(question, nn, passages, ft_model, sp)
    context = " ".join(retrieved_passages)

    # Check if the question is relevant to the Sri Lankan Constitution
    if not is_relevant_question(question, retrieved_passages, similarity_scores):
        return OUT_OF_SCOPE_MESSAGE, context, similarity_scores

    prompt = (
        f"Instructions: Use the provided context to answer the question accurately and completely in Sinhala. "
        f"Include specific references to articles or sections of the Sri Lankan Constitution when relevant. "
        f"If the context is insufficient to answer the specific question about the Sri Lankan Constitution, "
        f"respond with 'සමාවන්න, මට මෙම ප්‍රශ්නයට පිළිතුරු දීමට අවශ්‍ය තොරතුරු නොමැත.' (Sorry, I don't have the necessary information to answer this question.)\n"
        f"Context: {context}\n"
        f"Q: {question}\n"
        f"A:"
    )
    return generate_answer(prompt), context, similarity_scores


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
    answer, context, similarity_scores = answer_with_rag(question, nn, passages, ft_model, sp)
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