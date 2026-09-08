"""
RAG Starter Project (Groq + free local embeddings)
----------------------------------------------------------
Install first:
    pip install groq chromadb python-dotenv sentence-transformers

Add a .env file in the same folder with:
    GROQ_API_KEY=gsk_...

Get a free Groq key at https://console.groq.com
"""

import os
from dotenv import load_dotenv
import chromadb
from groq import Groq
from sentence_transformers import SentenceTransformer

# ---------- STEP 1: Setup & config ----------
load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# This downloads a small (~80MB) free embedding model the first time you run it.
embedder = SentenceTransformer("all-MiniLM-L6-v2")

CHAT_MODEL = "openai/gpt-oss-120b"  # current fast model on Groq's free tier
TOP_K = 3

# ---------- STEP 2: Load raw data ----------
NOTES_FILE = "about_me.txt"
def load_notes(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
        #split into paragraphswherever there is a blank line
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        return paragraphs
raw_paragraphs = load_notes(NOTES_FILE)
# ---------- STEP 3: Chunk the data ----------

def chunk_text(paragraphs, max_chunk_size=500,overlap = 30):
    all_words = " ". join(paragraphs).split()
    chunks = []
    start = 0
    while start < len(all_words):
        end = start + max_chunk_size
        chunks.append(" ". join(all_words[start:end]))
        start += max_chunk_size - overlap
    return chunks 

chunks = chunk_text(raw_paragraphs)


# ---------- STEP 4: Embed & store in a vector database ----------
'''
chroma is a vector database 
a special database that store list of number （embedding) 
and search through them by comparing the closeness in meaning 
not like nomrmal database (Mysql)
'''
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="study_notes")

def embed_text(text):
    return embedder.encode(text).tolist()

for i, chunk in enumerate(chunks):
    collection.add(
        ids=[str(i)],
        embeddings=[embed_text(chunk)],
        documents=[chunk],
    )

# ---------- STEP 5: Build the retriever ----------
'''this is the retriever function,
it will take the query and return the most relevant chunks from the vector
'''
def retrieve(query, k=TOP_K):
    query_embedding = embed_text(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=k)
    return results["documents"][0]

# ---------- STEP 6: Construct the augmented prompt ----------
'''this is where you define the prompt 
it will send to the llm to get the most closest answer
from the vector database'''
def build_prompt(query, retrieved_chunks):
    context = "\n\n".join(retrieved_chunks)
    return f"""Answer the question using ONLY the context below.
If the context doesn't contain the answer, say "I don't have that in my notes."

Context:
{context}

Question: {query}
Answer:"""

# ---------- STEP 7: Call the LLM (Groq) ----------
'''connect to the llm to get the answer from the prompt'''
def generate_answer(prompt):
    response = groq_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content

# ---------- STEP 8: Return the response ----------
'''send the response back to the user'''
def ask(query):
    retrieved = retrieve(query)
    prompt = build_prompt(query, retrieved)
    answer = generate_answer(prompt)
    print("Q:", query)
    print("A:", answer)
    print("\n(Retrieved from:", retrieved, ")")

if __name__ == "__main__":
    print("\nAsk a question (type 'exit' or 'quit' to stop):\n")
    while True:
        user_question = input("Q: ").strip()
        if user_question.lower() in ("exit", "quit"):
            break
        if user_question:
            ask(user_question)
