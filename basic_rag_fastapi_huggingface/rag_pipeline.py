from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.document_loaders import TextLoader
from langchain_community.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

# --- Step 1: Load and split documents ---
def create_vectorstore():
    loader = TextLoader("data/sample.txt")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    # --- Step 2: Create embeddings ---
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # --- Step 3: Store in Chroma ---
    vectorstore = Chroma.from_documents(chunks, embedding=embeddings, persist_directory="chroma_db")
    vectorstore.persist()
    return vectorstore


# --- Step 4: Create RAG chain ---
def create_rag_chain():
    # Load persisted Chroma DB
    vectorstore = Chroma(persist_directory="chroma_db", embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Load local LLM (can replace with OpenAI, Mistral, etc.)
    generator = pipeline(
        "text-generation",
        model="distilgpt2",
        max_new_tokens=200,
        temperature=0.3,
        do_sample=True,
    )

    llm = HuggingFacePipeline(pipeline=generator)

    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"  # simplest type
    )

    return rag_chain
