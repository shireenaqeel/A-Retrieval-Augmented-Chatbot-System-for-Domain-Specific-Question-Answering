import os
import streamlit as st
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from detoxify import Detoxify
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk

from groq import Groq

import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt_tab')


# ------------------ CONFIG ------------------ #
st.set_page_config(page_title="LLM QA with Full Metrics", layout="wide")
st.title("🧪 Web Q&A with Groq + Full Evaluation")

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")  # Set this in Streamlit Cloud > Settings > Secrets

# ------------------ FUNCTIONS ------------------ #
def get_website_content(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        return ' '.join(p.get_text() for p in paragraphs)
    except Exception as e:
        return f"Error: {e}"

def chunk_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def embed_chunks(chunks, model):
    return model.encode(chunks)

def create_faiss_index(embeddings):
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings))
    return index

def get_top_k_chunks(query, chunks, model, index, k=5):
    query_vec = model.encode([query])
    D, I = index.search(np.array(query_vec), k)
    top_chunks = [chunks[i] for i in I[0]]
    return top_chunks, D[0], I[0], query_vec[0]

def answer_question(question, context):
    client = Groq(api_key=GROQ_API_KEY)
    prompt = f"""You are a helpful assistant. Use the following context to answer the user's question.

Context:
{context}

Question:
{question}

Answer:"""
    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Groq Error: {e}"

def compute_bleu(answer, reference):
    answer_tokens = nltk.word_tokenize(answer)
    ref_tokens = [nltk.word_tokenize(reference)]
    return sentence_bleu(ref_tokens, answer_tokens, smoothing_function=SmoothingFunction().method1)

def compute_toxicity(text):
    try:
        results = Detoxify('original').predict(text)
        return results['toxicity']
    except:
        return None

# ------------------ UI ------------------ #
url = st.text_input("🌐 Enter a website URL")
question = st.text_input("❓ Ask a question about the webpage content")
reference_answer = st.text_input("📘 (Optional) Reference answer (for BLEU)")

if st.button("Submit") and url and question:
    with st.spinner("Scraping webpage..."):
        content = get_website_content(url)
        if not content or content.startswith("Error"):
            st.error(content)
            st.stop()

    with st.spinner("Embedding content..."):
        model = SentenceTransformer("all-MiniLM-L6-v2")
        chunks = chunk_text(content)
        embeddings = embed_chunks(chunks, model)
        index = create_faiss_index(embeddings)

    with st.spinner("Retrieving top chunks..."):
        top_chunks, distances, indices, query_vec = get_top_k_chunks(question, chunks, model, index)
        context = "\n\n".join(top_chunks)

        st.subheader("📚 Top Retrieved Chunks")
        for i, c in enumerate(top_chunks):
            st.markdown(f"**Chunk {i+1} (Distance {distances[i]:.4f}):**")
            st.write(c)

        st.subheader("📊 Retrieval Metrics")
        st.metric("Average L2 Distance", f"{np.mean(distances):.4f}")
        cosine_scores = cosine_similarity([query_vec], [model.encode([c])[0] for c in top_chunks])[0]
        st.metric("Average Cosine Similarity", f"{np.mean(cosine_scores):.4f}")

        # Visualize embeddings
        pca = PCA(n_components=2).fit_transform([query_vec] + [model.encode([c])[0] for c in top_chunks])
        plt.figure()
        sns.scatterplot(x=pca[:,0], y=pca[:,1], hue=["Query"] + [f"Chunk {i+1}" for i in range(len(top_chunks))])
        st.pyplot(plt)

        if st.checkbox("🔍 Show t-SNE Visualization"):
            tsne = TSNE(n_components=2, perplexity=5).fit_transform([query_vec] + [model.encode([c])[0] for c in top_chunks])
            plt.figure()
            sns.scatterplot(x=tsne[:,0], y=tsne[:,1], hue=["Query"] + [f"Chunk {i+1}" for i in range(len(top_chunks))])
            st.pyplot(plt)

    with st.spinner("Getting answer from LLM..."):
        answer = answer_question(question, context)

    st.subheader("📘 Answer")
    st.write(answer)

    st.subheader("🧠 Post-LLM Metrics")
    answer_len = len(answer.split())
    st.metric("Answer Length (words)", answer_len)
    context_vec = model.encode([context])[0]
    answer_vec = model.encode([answer])[0]
    ans_ctx_sim = cosine_similarity([context_vec], [answer_vec])[0][0]
    st.metric("Answer–Context Cosine Similarity", f"{ans_ctx_sim:.2f}")

    if reference_answer:
        bleu = compute_bleu(answer, reference_answer)
        st.metric("BLEU Score", f"{bleu:.2f}")

    tox = compute_toxicity(answer)
    if tox is not None:
        st.metric("Toxicity Score", f"{tox:.2f}")
    else:
        st.write("⚠️ Could not compute toxicity.")

    st.subheader("🗳️ Feedback")
    feedback = st.radio("Was this answer helpful?", ["👍 Yes", "👎 No"])

# ------------------ Footer ------------------ #
st.markdown("---")
st.caption("Built with ❤️ using Streamlit, SentenceTransformers, FAISS, and Groq")
