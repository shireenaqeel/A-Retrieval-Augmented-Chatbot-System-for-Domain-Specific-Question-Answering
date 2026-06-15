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
import pandas as pd
from rouge_score import rouge_scorer
import textstat

from groq import Groq

# Ensure required NLTK tokenizer data is available (newer NLTK needs 'punkt_tab')
@st.cache_resource
def _ensure_nltk_data():
    for pkg in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except LookupError:
            nltk.download(pkg)

_ensure_nltk_data()


# ------------------ CONFIG ------------------ #
st.set_page_config(page_title="LLM QA with Full Metrics", layout="wide")
st.title("🧪 Web Q&A with Groq + Full Evaluation")

# It is recommended to use st.secrets for API keys
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")  # Set this in Streamlit Cloud > Settings > Secrets

# ------------------ FUNCTIONS ------------------ #
@st.cache_data
def get_website_content(url):
    """Scrapes the text content from a website URL."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/124.0 Safari/537.36"
        }
        response = requests.get(url, timeout=10, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        return ' '.join(p.get_text() for p in paragraphs)
    except requests.exceptions.RequestException as e:
        return f"Error fetching URL: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

def chunk_text(text, chunk_size=500):
    """Splits text into smaller chunks."""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

@st.cache_resource
def load_embedding_model():
    """Loads the SentenceTransformer model."""
    return SentenceTransformer("all-MiniLM-L6-v2")

def embed_chunks(chunks, model):
    """Embeds text chunks using the SentenceTransformer model."""
    return model.encode(chunks)

def create_faiss_index(embeddings):
    """Creates a FAISS index for efficient similarity search."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index

def get_top_k_chunks(query, chunks, model, index, k=5):
    """Retrieves the top k most relevant chunks for a given query."""
    query_vec = model.encode([query])
    distances, indices = index.search(np.array(query_vec), k)
    top_chunks = [chunks[i] for i in indices[0]]
    return top_chunks, distances[0], indices[0], query_vec[0]

def answer_question(question, context):
    """Generates an answer using the Groq API based on the provided context."""
    if not GROQ_API_KEY or GROQ_API_KEY == "YOUR_GROQ_API_KEY":
        return "Error: Groq API key not configured. Please add your key to the script."
    client = Groq(api_key=GROQ_API_KEY)
    prompt = f"""You are a helpful assistant. Use the following context to answer the user's question.
If the context does not contain the answer, state that you cannot find the answer in the provided text.

Context:
{context}

Question:
{question}

Answer:"""
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # Current supported Groq model
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Groq Error: {e}"

def compute_bleu(answer, reference):
    """Computes the BLEU score between a generated answer and a reference."""
    answer_tokens = nltk.word_tokenize(answer)
    ref_tokens = [nltk.word_tokenize(reference)]
    return sentence_bleu(ref_tokens, answer_tokens, smoothing_function=SmoothingFunction().method1)

def compute_rouge(answer, reference):
    """Computes ROUGE scores (R-1, R-2, R-L)."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, answer)
    return {key: value.fmeasure for key, value in scores.items()}

def compute_toxicity(text):
    """Computes the toxicity score of a text."""
    try:
        results = Detoxify('original').predict(text)
        return results['toxicity']
    except Exception as e:
        st.warning(f"Could not compute toxicity: {e}")
        return None

def compute_readability(text):
    """Computes the Flesch-Kincaid Grade Level of a text."""
    return textstat.flesch_kincaid_grade(text)

# ------------------ UI ------------------ #
url = st.text_input("🌐 Enter a website URL", "https://en.wikipedia.org/wiki/Large_language_model")
question = st.text_input("❓ Ask a question about the webpage content", "What is a large language model?")
reference_answer = st.text_input("📘 (Optional) Provide a reference answer for BLEU/ROUGE scoring", "A large language model is a type of language model notable for its large size, trained on vast quantities of text using self-supervised learning.")

if st.button("Submit") and url and question:
    # --- 1. Content Fetching and Processing ---
    with st.spinner("Scraping webpage..."):
        content = get_website_content(url)
        if not content or content.startswith("Error"):
            st.error(content)
            st.stop()
    
    with st.spinner("Embedding content and building FAISS index..."):
        model = load_embedding_model()
        chunks = chunk_text(content)
        if not chunks:
            st.error("Could not extract any text content from the URL.")
            st.stop()
        embeddings = embed_chunks(chunks, model)
        index = create_faiss_index(embeddings)

    # --- 2. Retrieval ---
    with st.spinner("Retrieving relevant context..."):
        top_chunks, distances, indices, query_vec = get_top_k_chunks(question, chunks, model, index, k=5)
        context = "\n\n".join(top_chunks)
        chunk_embeddings = model.encode(top_chunks)
        cosine_scores = cosine_similarity([query_vec], chunk_embeddings)[0]

    st.subheader("🔍 Retrieval Evaluation")
    with st.expander("Show Retrieved Chunks and Relevance Metrics", expanded=True):
        chunk_data = {
            "Chunk": [f"Chunk {i+1}" for i in range(len(top_chunks))],
            "Text": top_chunks,
            "L2 Distance": distances,
            "Cosine Similarity": cosine_scores
        }
        df_chunks = pd.DataFrame(chunk_data)
        st.dataframe(df_chunks)

        # Plot Similarity Scores
        fig, ax = plt.subplots()
        sns.barplot(x="Chunk", y="Cosine Similarity", data=df_chunks, ax=ax, hue="Chunk", palette="viridis", legend=False)
        ax.set_title("Cosine Similarity of Retrieved Chunks to Query")
        st.pyplot(fig)

    # --- 3. Generation ---
    with st.spinner("Generating answer with Groq LLM..."):
        answer = answer_question(question, context)

    st.subheader("💬 Generated Answer")
    st.markdown(f"> {answer}")

    # --- 4. Post-LLM Evaluation ---
    st.subheader("📊 Generation & Evaluation Metrics")
    with st.expander("Show Detailed Evaluation Metrics & Graphs", expanded=True):
        # Prepare data for metrics calculation
        metrics_data = {}
        scalar_metrics = {}
        
        scalar_metrics["Answer Length (words)"] = len(answer.split())
        scalar_metrics["Readability (FK Grade)"] = compute_readability(answer)
        
        context_vec = model.encode([context])[0]
        answer_vec = model.encode([answer])[0]
        groundedness_score = cosine_similarity([context_vec], [answer_vec])[0][0]
        metrics_data["Groundedness"] = groundedness_score
        scalar_metrics["Groundedness Score"] = f"{groundedness_score:.4f}"

        tox = compute_toxicity(answer)
        metrics_data["Toxicity"] = tox if tox is not None else 0
        scalar_metrics["Toxicity Score"] = f"{tox:.4f}" if tox is not None else "N/A"

        if reference_answer:
            bleu_score = compute_bleu(answer, reference_answer)
            rouge_scores = compute_rouge(answer, reference_answer)
            ref_answer_vec = model.encode([reference_answer])[0]
            semantic_sim = cosine_similarity([answer_vec], [ref_answer_vec])[0][0]
            
            metrics_data.update({
                "BLEU": bleu_score,
                "ROUGE-1": rouge_scores['rouge1'],
                "ROUGE-2": rouge_scores['rouge2'],
                "ROUGE-L": rouge_scores['rougeL'],
                "Semantic Sim.": semantic_sim
            })
            scalar_metrics.update({
                "BLEU Score": f"{bleu_score:.4f}",
                "ROUGE-1": f"{rouge_scores['rouge1']:.4f}",
                "ROUGE-2": f"{rouge_scores['rouge2']:.4f}",
                "ROUGE-L": f"{rouge_scores['rougeL']:.4f}",
                "Semantic Answer Similarity": f"{semantic_sim:.4f}"
            })

        # Display Metrics in a Table
        st.write("#### Scalar Metrics")
        st.table(pd.DataFrame.from_dict(scalar_metrics, orient='index', columns=['Score']))
        
        # --- NEW: Additional Graphs ---
        st.write("#### Metric Visualizations")
        
        if reference_answer:
            col1, col2 = st.columns(2)
            
            # Graph 1: ROUGE Scores
            with col1:
                rouge_df = pd.DataFrame({
                    'Metric': ['ROUGE-1', 'ROUGE-2', 'ROUGE-L'],
                    'F1-Score': [metrics_data['ROUGE-1'], metrics_data['ROUGE-2'], metrics_data['ROUGE-L']]
                })
                fig_rouge, ax_rouge = plt.subplots()
                sns.barplot(data=rouge_df, x='Metric', y='F1-Score', ax=ax_rouge, hue='Metric', palette='plasma', legend=False)
                ax_rouge.set_title('ROUGE F1-Scores')
                ax_rouge.set_ylim(0, 1)
                st.pyplot(fig_rouge)

            # Graph 2: Similarity Comparison
            with col2:
                sim_df = pd.DataFrame({
                    'Metric': ['Groundedness', 'Semantic Similarity'],
                    'Score': [metrics_data['Groundedness'], metrics_data['Semantic Sim.']]
                })
                fig_sim, ax_sim = plt.subplots()
                sns.barplot(data=sim_df, x='Metric', y='Score', ax=ax_sim, hue='Metric', palette='coolwarm', legend=False)
                ax_sim.set_title('Similarity Scores')
                ax_sim.set_ylabel('Cosine Similarity')
                ax_sim.set_ylim(0, 1)
                st.pyplot(fig_sim)

            # Graph 3: Radar Chart for Overall Quality
            st.write("##### Overall Answer Quality Profile")
            radar_labels = ['Groundedness', 'Semantic Sim.', 'BLEU', 'ROUGE-L', 'Safety (1-Tox)']
            radar_values = [
                metrics_data.get('Groundedness', 0),
                metrics_data.get('Semantic Sim.', 0),
                metrics_data.get('BLEU', 0),
                metrics_data.get('ROUGE-L', 0),
                1 - metrics_data.get('Toxicity', 0)
            ]
            
            angles = np.linspace(0, 2 * np.pi, len(radar_labels), endpoint=False).tolist()
            radar_values += radar_values[:1]
            angles += angles[:1]

            fig_radar, ax_radar = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
            ax_radar.fill(angles, radar_values, color='teal', alpha=0.25)
            ax_radar.plot(angles, radar_values, color='teal', linewidth=2)
            ax_radar.set_yticklabels([])
            ax_radar.set_xticks(angles[:-1])
            ax_radar.set_xticklabels(radar_labels)
            ax_radar.set_title("Answer Quality Radar", size=15, color='teal', y=1.1)
            st.pyplot(fig_radar)

    st.subheader("🗳️ Feedback")
    st.radio("Was this answer helpful?", ["👍 Yes", "👎 No"], key="feedback")

# ------------------ Footer ------------------ #
st.markdown("---")
st.caption("Built with ❤️ using Streamlit, SentenceTransformers, FAISS, and Groq")
