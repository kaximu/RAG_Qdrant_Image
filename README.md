💬 RAG_Qdrant_Image

A Retrieval-Augmented Generation (RAG) chatbot built with Streamlit, supporting:

🌍 Website ingestion (text + OCR on images)

📂 File uploads (TXT, PDF, PNG, JPG)

🖼️ Scanned PDFs & images (OCR with EasyOCR)

🧠 Qdrant / FAISS vector search + hybrid BM25 retrieval

⚡ Multi-query expansion & reranking for better answers

📊 Admin dashboard with feedback logging, performance charts, and question analysis

🚀 Features

OCR with EasyOCR → works on scanned PDFs, images, and website screenshots

Hybrid retrieval → dense (Qdrant / FAISS) + sparse (BM25)

LLM integration:

Uses OpenAI GPT if OPENAI_API_KEY is available

Falls back to HuggingFace FLAN-T5 if not

Admin dashboard:

Feedback logging 👍👎

Accuracy & speed comparison

Similarity vs. feedback scatterplots

Export results as JSON/TXT
🛠️ Tech Stack

Frontend: Streamlit

Vector stores: Qdrant Cloud, FAISS (local)

Retrieval: LangChain, BM25, EnsembleRetriever

OCR: EasyOCR + pdf2image

Embeddings: Sentence-Transformers, OpenAI embeddings

LLMs: OpenAI GPT, HuggingFace FLAN-T5

📖 Example Use Cases

Build knowledge bots from company websites (including images/infographics)

Index scanned reports, research papers, and forms

Provide hybrid search across structured + unstructured content

✅ Ready for deployment on Streamlit.io 🚀