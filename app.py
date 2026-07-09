import os
import zipfile
import requests
import chromadb
from flask import Flask, request, jsonify, render_template_string

###########################################################
# Configuration
###########################################################
NVIDIA_API_KEY = os.environ["NVIDIA_API_KEY"]
EMBEDDINGS_URL = "https://integrate.api.nvidia.com/v1/embeddings"
CHAT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

EMBEDDING_MODEL = "nvidia/nv-embedqa-e5-v5"
CHAT_MODEL = "meta/llama-3.1-8b-instruct"

CHROMA_DB = "./course_kb"
CHROMA_ZIP = "./course_kb.zip"
TOP_K = 4

###########################################################
# Auto-unzip the knowledge base on first startup
###########################################################
if not os.path.isdir(CHROMA_DB) and os.path.isfile(CHROMA_ZIP):
    print(f"Extracting {CHROMA_ZIP} ...")
    with zipfile.ZipFile(CHROMA_ZIP, "r") as zf:
        zf.extractall(".")
    print("Extraction complete.")

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
        "input_type": "query",
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
    return response.json()["choices"][0]["message"]["content"]


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
# Flask app
###########################################################
app = Flask(__name__)

PAGE_HTML = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SyllabusBot</title>
  <style>
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif;
      background: #0f1420;
      color: #e8ecf4;
      display: flex;
      flex-direction: column;
      height: 100vh;
    }
    header {
      padding: 18px 20px;
      background: #141a2b;
      border-bottom: 1px solid #232a3d;
    }
    header h1 { margin: 0; font-size: 18px; }
    header p { margin: 4px 0 0; font-size: 13px; color: #8b93a7; }
    #chat {
      flex: 1;
      overflow-y: auto;
      padding: 20px;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }
    .msg { max-width: 75%; padding: 10px 14px; border-radius: 12px; line-height: 1.45; font-size: 14px; }
    .user { align-self: flex-end; background: #2563eb; color: white; }
    .bot { align-self: flex-start; background: #1c2338; border: 1px solid #2b334a; }
    .source { font-size: 11px; color: #8b93a7; margin-top: 6px; font-style: italic; }
    #inputBar { display: flex; padding: 14px; gap: 10px; border-top: 1px solid #232a3d; background: #141a2b; }
    #question { flex: 1; padding: 12px 14px; border-radius: 10px; border: 1px solid #2b334a; background: #0f1420; color: #e8ecf4; font-size: 14px; }
    #question:focus { outline: none; border-color: #2563eb; }
    button { padding: 12px 20px; border-radius: 10px; border: none; background: #2563eb; color: white; font-size: 14px; cursor: pointer; }
    button:disabled { opacity: 0.5; cursor: default; }
    .loading { color: #8b93a7; font-style: italic; }
  </style>
</head>
<body>
  <header>
    <h1>SyllabusBot</h1>
    <p>Ask a question about your course material — answers are grounded in your syllabus only.</p>
  </header>
  <div id="chat"></div>
  <div id="inputBar">
    <input id="question" type="text" placeholder="Ask a question..." autocomplete="off">
    <button id="sendBtn" onclick="sendQuestion()">Send</button>
  </div>

  <script>
    const chat = document.getElementById('chat');
    const input = document.getElementById('question');
    const btn = document.getElementById('sendBtn');

    function addMessage(text, sender, source) {
      const div = document.createElement('div');
      div.className = 'msg ' + sender;
      div.innerText = text;
      if (source) {
        const src = document.createElement('div');
        src.className = 'source';
        src.innerText = 'Source: ' + source;
        div.appendChild(src);
      }
      chat.appendChild(div);
      chat.scrollTop = chat.scrollHeight;
      return div;
    }

    async function sendQuestion() {
      const question = input.value.trim();
      if (!question) return;
      addMessage(question, 'user');
      input.value = '';
      btn.disabled = true;

      const loadingMsg = addMessage('Thinking...', 'bot');
      loadingMsg.classList.add('loading');

      try {
        const res = await fetch('/ask', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question })
        });
        const data = await res.json();
        loadingMsg.remove();
        addMessage(data.answer || 'Something went wrong.', 'bot', data.source);
      } catch (e) {
        loadingMsg.remove();
        addMessage('Error reaching the server. Please try again.', 'bot');
      }
      btn.disabled = false;
    }

    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') sendQuestion();
    });
  </script>
</body>
</html>
"""


@app.route("/")
def home():
    return render_template_string(PAGE_HTML)


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True)
    question = (data.get("question") or "").strip()

    if not question:
        return jsonify({"answer": "Please type a question."})

    try:
        chunks = retrieve(question)
    except Exception as e:
        return jsonify({"answer": f"Retrieval error: {e}"})

    if not chunks:
        return jsonify({"answer": "I couldn't find anything relevant in the course material."})

    try:
        answer = ask_llm(question, chunks)
    except Exception as e:
        return jsonify({"answer": f"Answer generation error: {e}"})

    source = chunks[0]["filename"] if chunks else None
    return jsonify({"answer": answer, "source": source})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
