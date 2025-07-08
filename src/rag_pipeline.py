from sentence_transformers import SentenceTransformer
import chromadb
import numpy as np
from transformers import pipeline
from huggingface_hub import login




# === Load embedding model and vector store ===
model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="../data/vector_store")
collection = client.get_or_create_collection(name="complaints")

# === Retriever Function ===
def retrieve_chunks(query, k=3):
    query_embedding = model.encode(query).tolist()
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


# Use a lightweight LLM (e.g., distilgpt2, mistral, llama2) or local model
#qa_pipeline = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.1")
#qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-small")
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")





def generate_answer(prompt):
    response = qa_pipeline(prompt, max_new_tokens=100, do_sample=False)
    return response[0]["generated_text"].strip()


def rag_pipeline(question, k=5):
    try:
        chunks, metadatas = retrieve_chunks(question, k)
        prompt = build_prompt(chunks, question)
        answer = generate_answer(prompt)
        return answer, chunks[:2]
    except Exception as e:
        print("Error inside rag_pipeline:", e)
        return None, []
print("Script started")
questions = [
    "What common issues are reported with credit cards?",
    "Why are customers unhappy with Buy Now, Pay Later?",
    "What common issues are reported with credit cards?",
    "Are there any frequent complaints about money transfers?",
    "What are customers saying about savings accounts?",
    "Why do people file complaints about personal loans?"
]

for q in questions:
    print(f"\n📌 Question: {q}")
    answer, sources = rag_pipeline(q)
    print(f"\n🧠 Answer:\n{answer}")
    #print("Source 1:", sources[1][:200])
    print(f"\n📄 source 1 (Preview):\n- {sources[0][:200]}...\n- {sources[1][:200]}...\n")
    
print("Script ended")






