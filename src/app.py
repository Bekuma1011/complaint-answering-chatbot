import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import chromadb

# === Load Models and Vector Store ===
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")

client = chromadb.PersistentClient(path="../data/vector_store")
collection = client.get_or_create_collection(name="complaints")

# === RAG Functions ===
def retrieve_chunks(query, k=3):
    query_embedding = embedding_model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )
    return results["documents"][0], results["metadatas"][0]

def build_prompt(context_chunks, question):
    context = "\n\n".join(context_chunks)
    prompt = f"""
You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints.

Use the following retrieved complaint excerpts to formulate your answer. If the context doesn't contain the answer, state that you don't have enough information.

Context:
{context}

Question:
{question}

Answer:"""
    return prompt

def generate_answer(prompt):
    response = qa_pipeline(prompt, max_new_tokens=100, do_sample=False)
    return response[0]["generated_text"].strip()

def rag_pipeline(question, k=5):
    chunks, metadatas = retrieve_chunks(question, k)
    prompt = build_prompt(chunks, question)
    answer = generate_answer(prompt)
    return answer, chunks

# === Streamlit Interface ===
st.set_page_config(page_title="CrediTrust Complaint Chatbot", page_icon="💬")

st.title("💬 CrediTrust Complaint Insights")
st.markdown("Ask a question about customer complaints across products like credit cards, Personal loan, Buy Now Pay Later, Savings account, Money transfer")

with st.form(key="question_form"):
    user_input = st.text_input("Enter your question:", placeholder="e.g., Why are people unhappy with BNPL?")
    submit = st.form_submit_button("Ask")

if submit and user_input.strip():
    with st.spinner("Thinking..."):
        answer, sources = rag_pipeline(user_input)

    st.subheader(" Answer")
    st.write(answer)

    st.subheader("Sources")
    for i, src in enumerate(sources):
        st.markdown(f"**Source {i+1}:**")
        st.markdown(f"> {src[:400]}...")  # Truncate long source texts

# Clear button (optional)
if st.button("Clear"):
    st.experimental_rerun()
