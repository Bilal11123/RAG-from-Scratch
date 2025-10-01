# 📚 RAG Pipeline with OpenAI & Gemini

This repository contains implementations of a **Retrieval-Augmented Generation (RAG) pipeline** using [LangChain](https://www.langchain.com/) and [Chroma](https://www.trychroma.com/).  
It supports **two backends** for the LLM and embeddings:

- 🔹 **OpenAI (ChatGPT + OpenAI Embeddings)**  
- 🔹 **Google Gemini API (Generative AI + Gemini Embeddings)**  

The pipeline can ingest documents from **web pages** and **local PDF files**, index them into a vectorstore (Chroma), and answer natural language questions using retrieved context.
### Uses LangSmith: https://docs.smith.langchain.com/
### Credit to: [RAG quickstart](https://python.langchain.com/docs/tutorials/rag/)


---

## 🚀 Features

- ✅ Web & PDF ingestion  
- ✅ Document chunking & embedding  
- ✅ Persistent Chroma vector database  
- ✅ Query answering with context  
- ✅ Works with **OpenAI** or **Gemini**  
- ✅ Easily extendable with more document sources  

---

## 📂 Project Structure

- `.` (repo root)
  - `openai_rag.py` — RAG pipeline with OpenAI (ChatGPT + OpenAIEmbeddings)
  - `gemini_rag.py` — RAG pipeline with Gemini (Generative AI + Gemini Embeddings)
  - `requirements.txt` — Python dependencies
  - `README.md` — Project documentation

---

## ⚙️ Setup

### 1️⃣ Clone Repository
```bash
git clone https://github.com/your-username/rag-pipeline.git
cd rag-pipeline
```

### 2️⃣ Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🔑 API Keys & Environment Variables

Both implementations require API keys. Store them in **environment variables** or a `.env` file.

### OpenAI
```bash
export OPENAI_API_KEY="your-openai-key"
```

### Gemini
```bash
export GOOGLE_API_KEY="your-gemini-key"
```

Or in a `.env` file:
```
OPENAI_API_KEY=your-openai-key
GOOGLE_API_KEY=your-gemini-key
```

> Tip: For local development you can use `python-dotenv` and `from dotenv import load_dotenv` to load `.env`.

---

## ▶️ Usage

### OpenAI Version
```bash
python openai_rag.py
```

### Gemini Version
```bash
python gemini_rag.py
```

Both scripts:
- Load docs into Chroma (`add_pdf()` or `add_url()`)  
- Retrieve relevant chunks  
- Query the LLM with context  
- Print the answer  

---

## 📝 Example (Python usage)

```python
# Add a PDF
add_pdf("myfile.pdf")

# Add a webpage
add_url("https://lilianweng.github.io/posts/2023-06-23-agent/")

# Ask a question
answer = ask_llm("What is Task Decomposition?")
print(answer)
```

---

## 📌 Roadmap

- [ ] Add streaming responses  
- [ ] Support more file formats (Word, TXT, CSV)  
- [ ] Expose as REST API with FastAPI  
- [ ] Add UI with Streamlit  

---

## 🤝 Contributing

Pull requests are welcome!  
For major changes, please open an issue first to discuss what you’d like to change.  
