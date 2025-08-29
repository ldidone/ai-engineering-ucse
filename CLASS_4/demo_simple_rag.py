import time
import ollama
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# load environment variables from .env file
load_dotenv()
client = genai.Client()

# Load the dataset
dataset = []
script_dir = os.path.dirname(__file__)
with open(os.path.join(script_dir, "data", "cat-facts.txt"), "r") as file:
    dataset = file.readlines()
    print(f"Loaded {len(dataset)} entries")


# Implement the retrieval system
EMBEDDING_MODEL = "hf.co/CompendiumLabs/bge-base-en-v1.5-gguf"
LANGUAGE_MODEL = "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF"

# Embedding backend selection: 'auto' (decide at first call), or forced 'ollama'/'st'
embedding_backend = "auto"
st_model = None

# Use Gemini as a Language Model
USE_GEMINI = True

def ensure_st_model():
    global st_model
    if st_model is None:
        if SentenceTransformer is None:
            raise RuntimeError(
                "SentenceTransformers is not installed. Install with: pip install sentence-transformers"
            )
        st_model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_with_ollama(text):
    return ollama.embed(model=EMBEDDING_MODEL, input=text)["embeddings"][0]


def embed_with_st(text):
    ensure_st_model()
    # Ensure list[float]
    vec = st_model.encode(text)
    try:
        return vec.tolist()
    except AttributeError:
        return list(vec)


def get_embedding(text):
    global embedding_backend
    # If a backend was chosen already, stick to it for dimensional consistency
    if embedding_backend == "st":
        return embed_with_st(text)
    if embedding_backend == "ollama":
        return embed_with_ollama(text)

    # Auto select on first call: try SentenceTransformers first, fall back to Ollama
    start_time = time.time()
    try:
        emb = embed_with_st(text)
        elapsed = time.time() - start_time
        if elapsed > 25.0:
            # Switch to Ollama and re-embed to keep a single embedding space
            print(
                "Embedding via SentenceTransformers was slow (>1s). Switching to Ollama for this run."
            )
            embedding_backend = "ollama"
            return embed_with_ollama(text)
        else:
            embedding_backend = "st"
            return emb
    except Exception as exc:
        print(f"SentenceTransformers embedding failed ({exc}). Falling back to Ollama.")
        try:
            embedding_backend = "ollama"
            return embed_with_ollama(text)
        except Exception as oll_exc:
            # If both fail, raise a combined error
            raise RuntimeError(
                f"Both embedding backends failed: Ollama error: {oll_exc}"
            )


# Each element in the VECTOR_DB will be a tuple (chunk, embedding)
# The embedding is a list of floats, for example: [0.1, 0.04, -0.34, 0.21, ...]
VECTOR_DB = []

def add_chunk_to_database(chunk, index, total):
    try:
        print(f"Embedding chunk {index}/{total}...")
        embedding = get_embedding(chunk)
        VECTOR_DB.append((chunk, embedding))
        print(f"Embedded chunk {index}/{total} using {embedding_backend.upper()}")
    except Exception as e:
        print(f"Error adding chunk to the database: {e}")


for i, chunk in enumerate(dataset, start=1):
    print(f"Adding chunk {i}/{len(dataset)} to the database...")
    add_chunk_to_database(chunk, i, len(dataset))
    print(f"Added chunk {i}/{len(dataset)} successfully to the database")


def cosine_similarity(a, b):
    dot_product = sum([x * y for x, y in zip(a, b)])
    norm_a = sum([x**2 for x in a]) ** 0.5
    norm_b = sum([x**2 for x in b]) ** 0.5
    return dot_product / (norm_a * norm_b)


def retrieve(query, top_n=3):
    # Ensure query embedding is in the same space as the indexed chunks
    query_embedding = get_embedding(query)
    # temporary list to store (chunk, similarity) pairs
    similarities = []
    for chunk, embedding in VECTOR_DB:
        similarity = cosine_similarity(query_embedding, embedding)
        similarities.append((chunk, similarity))
    # sort by similarity in descending order, because higher similarity means more relevant chunks
    similarities.sort(key=lambda x: x[1], reverse=True)
    # finally, return the top N most relevant chunks
    return similarities[:top_n]


# Chatbot
while True:
  input_query = input("Ask me a question (enter 'exit' to quit): ")
  if input_query.lower() == "exit":
      break

  retrieved_knowledge = retrieve(input_query)

  print("Retrieved knowledge:")
  for chunk, similarity in retrieved_knowledge:
      print(f" - (similarity: {similarity:.2f}) {chunk}")

  context_lines = "\n".join(
      [f" - {chunk.strip()}" for chunk, similarity in retrieved_knowledge]
  )
  instruction_prompt = f"""You are a helpful chatbot.
  Use only the following pieces of context to answer the question. Don't make up any new information:
  {context_lines}
  """
  # print(instruction_prompt)

  if USE_GEMINI:
      response = client.models.generate_content(
          model="gemini-2.5-flash",
          config=types.GenerateContentConfig(
              system_instruction=instruction_prompt,
          ),
          contents=input_query,
      )
      # print the response from the chatbot
      print(f"Chatbot response: {response.text}")
  else:
      stream = ollama.chat(
          model=LANGUAGE_MODEL,
          messages=[
              {"role": "system", "content": instruction_prompt},
              {"role": "user", "content": input_query},
          ],
          stream=True,
      )

      # print the response from the chatbot in real-time
      print("Chatbot response:")
      for chunk in stream:
          print(chunk["message"]["content"], end="", flush=True)
