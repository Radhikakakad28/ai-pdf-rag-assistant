AI PDF Q&A Assistant (RAG System – No API Key Needed)

This project is a fully free AI PDF Question-Answering system built using:

LangChain

FAISS Vector Store

HuggingFace MiniLM Embeddings

Qwen 2.5 1.5B Instruct LLM (Open & Free)

Python & Google Colab

The system allows you to upload any PDF and ask questions. It reads the document, converts it into vector embeddings, retrieves relevant text, and generates accurate answers using a Retrieval-Augmented Generation (RAG) pipeline.

🚀 Features

✔ Upload any PDF and chat with it

✔ 100% Free – No OpenAI key required

✔ Works on Google Colab

✔ Uses FAISS for fast similarity search

✔ Uses open-source LLM (Qwen) for answer generation

✔ Summarization, explanation, Q&A, MCQs, etc.

🧰 Tech Stack

Python

LangChain

FAISS

Sentence-Transformers

HuggingFace Transformers

Qwen 2.5 LLM

Google Colab

📦 Installation (Google Colab)
1️⃣ Install Dependencies
!pip install langchain langchain-community faiss-cpu pypdf sentence-transformers transformers accelerate bitsandbytes

2️⃣ Upload PDF
from google.colab import files
uploaded = files.upload()
pdf_path = list(uploaded.keys())[0]

3️⃣ Load & Split PDF
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = PyPDFLoader(pdf_path)
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = splitter.split_documents(docs)

4️⃣ Create Embeddings + FAISS Vector Store
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(chunks, embeddings)

5️⃣ Load Free LLM (Qwen 2.5)
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16
)

6️⃣ RAG Answer Function
def answer_query(query):
    docs = vector_store.similarity_search(query, k=3)
    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""Use the context below to answer the question.

Context:
{context}

Question: {query}

Answer:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=200, temperature=0.2)
    return tokenizer.decode(output[0], skip_special_tokens=True)

7️⃣ Ask a Question
answer_query("Give me a summary in 5 bullet points.")

📁 Project Structure
ai-pdf-rag-assistant/
│── pdf_qa.ipynb
│── README.md
│── requirements.txt (optional)

🎯 Use Cases

Summaries

Viva questions

Extract keywords

Chapter explanations

Notes generator

MCQs from PDF
