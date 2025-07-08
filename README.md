# Complaint Answering Chatbot – RAG Pipeline

This project implements a Retrieval-Augmented Generation (RAG) system for answering customer complaints using a large language model (LLM) and vector-based document retrieval. It is structured into modular tasks.

---

## Text Chunking, Embedding, and Vector Store Indexing


Convert long complaint narratives into manageable chunks, generate vector embeddings, and store them in a retrievable format.

### Key Steps
- **Text Chunking**: Implemented using `LangChain`’s `RecursiveCharacterTextSplitter` to break documents into semantically meaningful chunks.
  - Tuned `chunk_size` and `chunk_overlap` for balance.
- **Embedding**: Used a transformer-based embedding model (e.g., `all-MiniLM-L6-v2`).
- **Vector Store**: Stored vectors using `ChromaDB` or `FAISS` for efficient semantic search.

---

## Model Loading and RAG Pipeline Setup


Set up a RAG pipeline combining vector-based retrieval with a large language model for answer generation.

### Key Steps
- **Model Used**: [`mistralai/Mistral-7B-Instruct-v0.1`](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1) from Hugging Face.
- **Pipeline**:
  - Used `transformers.pipeline("text-generation", ...)`.
  - Retrieved relevant chunks from the vector store based on user query.
  - Passed context + query into LLM to generate a contextualized answer.



---

## Creating an Interactive Chat Interface

Build an easy-to-use frontend that allows non-technical users to interact with the chatbot.

###  Framework Used
- **Option 2**: [Streamlit](https://streamlit.io/)

### Core Features
- **Text Input**: Box for user to type questions.
- **Submit Button**: Triggers LLM-based response generation.
- **Answer Display**: Shows the generated answer.

### Enhancements
- **Display Sources**: Shows source text chunks used to generate the answer.
- **Streaming (Optional)**: Displays responses token-by-token for better user experience.
- **Clear Button**: Resets the chat window.

---

## 📦 Dependencies

```bash
pip install -r requirements.txt

---
## start app
streamlit run app.py
---
