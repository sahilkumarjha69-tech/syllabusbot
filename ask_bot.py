import os
import json
import requests
import chromadb

###########################################################
# Configuration
###########################################################
NVIDIA_API_KEY = os.environ["NVIDIA_API_KEY"]
EMBEDDINGS_URL = "https://integrate.api.nvidia.com/v1/embeddings"
CHAT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

EMBEDDING_MODEL = "nvidia/nv-embedqa-e5-v5"
CHAT_MODEL = "meta/llama-3.1-8b-instruct"  # fast, free-tier friendly chat model

CHROMA_DB = "./course_kb"
TOP_K = 4  # how many chunks to retrieve per question

###########################################################
# Chroma DB
###########################################################
client = chromadb.PersistentClient(path=CHROMA_DB)
collection = client.get_or_create_collection(name="course_materials")

###########################################################
# NVIDIA Embeddings (query mode)
###########################################################
def embed_query(text):
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {
        "input": [text],
        "model": EMBEDDING_MODEL,
        "input_type": "query",  # different from build_kb.py, which uses "passage"
        "encoding_format": "float",
    }
    response = requests.post(EMBEDDINGS_URL, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]


###########################################################
# NVIDIA Chat Completion
###########################################################
def ask_llm(question, context_chunks):
    context_text = "\n\n---\n\n".join(
        f"[Source: {c['filename']}]\n{c['text']}" for c in context_chunks
    )

    system_prompt = (
        "You are SyllabusBot, a course assistant. Answer the student's question "
        "using ONLY the provided course material below. If the answer isn't in "
        "the material, say so clearly instead of guessing. Keep answers concise "
        "and student-friendly. Cite the source filename at the end of your answer."
    )

    user_prompt = f"Course material:\n{context_text}\n\nStudent question: {question}"

    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 500,
    }

    response = requests.post(CHAT_URL, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


###########################################################
# Retrieval
###########################################################
def retrieve(question, top_k=TOP_K):
    query_embedding = embed_query(question)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    chunks = []
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    for doc, meta in zip(docs, metas):
        chunks.append({"text": doc, "filename": meta.get("filename", "unknown")})
    return chunks


###########################################################
# Main loop
###########################################################
def main():
    print("SyllabusBot (command-line test mode)")
    print("Type a question about your course material, or 'quit' to exit.\n")

    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in ("quit", "exit"):
            break

        try:
            chunks = retrieve(question)
        except Exception as e:
            print(f"  ! retrieval failed: {e}\n")
            continue

        if not chunks:
            print("SyllabusBot: I couldn't find anything relevant in the course material.\n")
            continue

        try:
            answer = ask_llm(question, chunks)
        except Exception as e:
            print(f"  ! answer generation failed: {e}\n")
            continue

        print(f"\nSyllabusBot: {answer}\n")


if __name__ == "__main__":
    main()