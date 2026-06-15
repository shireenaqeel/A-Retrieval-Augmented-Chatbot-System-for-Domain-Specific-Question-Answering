<h1 align="center">🧪 Web Q&A with Groq — RAG + Evaluation</h1>

<p align="center">
  Ask any question about a live web page and get an answer grounded in its content,
  <br/>built with Retrieval-Augmented Generation and a full suite of evaluation metrics.
</p>

<p align="center">
  <a href="https://ragforwebdomain.streamlit.app/">
    <img src="https://img.shields.io/badge/🚀%20Live%20Demo-Open%20App-7FB0E0?style=for-the-badge" alt="Live Demo" />
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/FAISS-0467DF?style=flat-square&logo=meta&logoColor=white" />
  <img src="https://img.shields.io/badge/Groq-F55036?style=flat-square&logo=groq&logoColor=white" />
  <img src="https://img.shields.io/badge/RAG-1C3C3C?style=flat-square" />
</p>

---

## 📖 Overview

This app takes the URL of any web page, reads its content, and lets you **ask questions about it in plain language**. Instead of guessing, the model answers using only the text retrieved from that page — the core idea behind **Retrieval-Augmented Generation (RAG)**.

On top of generating answers, it also **evaluates** them: how grounded the answer is, how readable, how toxic, and (if you give a reference answer) BLEU / ROUGE / semantic-similarity scores — all visualised with charts.

**▶️ Try it live:** [ragforwebdomain.streamlit.app](https://ragforwebdomain.streamlit.app/)

---

## ✨ Features

- 🌐 **Scrapes any web page** and extracts its readable text
- ✂️ **Chunks & embeds** the text using `sentence-transformers` (MiniLM)
- 🔍 **FAISS vector search** to retrieve the most relevant passages for your question
- 🤖 **Groq-powered LLM** generates a context-grounded answer
- 📊 **Built-in evaluation suite:**
  - Groundedness (answer vs. retrieved context)
  - Readability (Flesch–Kincaid grade)
  - Toxicity check (Detoxify)
  - BLEU, ROUGE-1/2/L & semantic similarity vs. a reference answer
- 📈 **Visual insights** — bar charts and a radar “answer quality” profile
- 🎨 Clean dark UI with a blue-pastel theme

---

## 🧠 How It Works

```
URL → Scrape text → Chunk → Embed (MiniLM) → FAISS index
                                                  │
                       Question ── embed ──→ retrieve top-k chunks
                                                  │
                          Context + Question → Groq LLM → Answer
                                                  │
                                       Evaluate & visualise metrics
```

---

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| **Frontend / App** | Streamlit |
| **Embeddings** | sentence-transformers (`all-MiniLM-L6-v2`) |
| **Vector search** | FAISS |
| **LLM** | Groq (`llama-3.1-8b-instant`) |
| **Scraping** | requests, BeautifulSoup |
| **Evaluation** | NLTK (BLEU), rouge-score, Detoxify, textstat |
| **Viz / Data** | matplotlib, seaborn, pandas, numpy |

---

## 🚀 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/shireenaqeel/A-Retrieval-Augmented-Chatbot-System-for-Domain-Specific-Question-Answering.git
cd A-Retrieval-Augmented-Chatbot-System-for-Domain-Specific-Question-Answering

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your Groq API key
#    Create a file: .streamlit/secrets.toml  with:
#    GROQ_API_KEY = "your_groq_api_key_here"

# 4. Run the app
streamlit run main.py
```

> Get a free Groq API key at [console.groq.com](https://console.groq.com). Never commit your key — `.streamlit/secrets.toml` is already in `.gitignore`.

---

## 📌 Note

`main.py` is the main app. The evaluation metrics make this more than a basic chatbot — it measures *how good* each answer actually is.

---

<p align="center">Made with 💙 by <a href="https://github.com/shireenaqeel">Shireen Ansari</a></p>
