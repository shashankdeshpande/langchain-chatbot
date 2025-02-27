import json
import faiss
import numpy as np
import ollama
import requests
import chromadb
from chromadb.utils import embedding_functions
import argparse
from tqdm import tqdm


def load_jsonl(file_path):
    """Load JSON Lines file as a generator."""
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)


def format_item_text(item):
    """Format structured item data into a single text representation."""
    return f"""
    Name: {item.get('name', 'N/A')}
    Description: {item.get('description', 'N/A')}
    Style: {item.get('style', 'N/A')}
    Category: {item.get('category', 'N/A')}
    """.strip()


def get_ollama_embedding(text):
    """Generate embeddings using Ollama."""
    response = ollama.embeddings(model='mxbai-embed-large', prompt=text)
    return response['embedding']

def get_mixedbread_embedding(text):
    """Generate embeddings using Mixedbread.ai."""
    url = "https://api.mixedbread.ai/v1/embeddings"
    headers = {"Authorization": "Bearer YOUR_API_KEY", "Content-Type": "application/json"}
    data = {"model": "mixedbread-1", "input": text}
    response = requests.post(url, headers=headers, json=data)
    return response.json()["data"][0]["embedding"]

def store_in_faiss(embeddings):
    """Store embeddings in FAISS index."""
    d = len(embeddings[0])  # Dimension of embeddings
    index = faiss.IndexFlatL2(d)
    index.add(np.array(embeddings, dtype='float32'))
    faiss.write_index(index, "embeddings.index")

def store_in_chromadb(items):
    """Store embeddings, item IDs, and URLs in ChromaDB."""
    client = chromadb.PersistentClient(path="chromadb_store")
    collection = client.get_or_create_collection(name="embeddings")
    for item in items:
        collection.add(ids=[str(item['id'])], embeddings=[item['embedding']], documents=[json.dumps(item)])

def search_chromadb(query_text, top_k=5):
    """Search ChromaDB for similar embeddings."""
    client = chromadb.PersistentClient(path="chromadb_store")
    collection = client.get_collection(name="embeddings")
    query_embedding = get_ollama_embedding(query_text)
    query_embedding = normalize_embedding(query_embedding)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    import epdb; epdb.st()
    return results["documents"][0] if "documents" in results else []


def normalize_embedding(embedding):
    """Apply L2 normalization to an embedding."""
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm > 0 else embedding


def update_chromadb_embeddings(fname):
    """Retrieve, normalize, and update embeddings in ChromaDB."""
    client = chromadb.PersistentClient(path=fname)
    collection = client.get_collection(name="embeddings")

    # Retrieve all stored embeddings
    results = collection.get()  # Retrieves all data (ids, embeddings, and documents)
    
    ids = results["ids"]
    embeddings = results["embeddings"]
    embeddings = [json.loads(doc)["embedding"] for doc in results["documents"]]
    documents = []
    for doc in results["documents"]:
        doc = json.loads(doc)
        doc.pop("embedding")
        documents.append(json.dumps(doc))

    # Normalize embeddings
    normalized_embeddings = [normalize_embedding(embedding) for embedding in embeddings]

    # Delete old embeddings
    collection.delete(ids=ids)

    # Reinsert normalized embeddings
    collection.add(ids=ids, embeddings=normalized_embeddings, documents=documents)

    print("ChromaDB embeddings updated with L2 normalization.")
    
def main():
    
    parser = argparse.ArgumentParser(description="Embedding storage and retrieval")
    parser.add_argument("--jsonl", type=str, required=True, help="Path to JSONL file")
    parser.add_argument("--db-name", type=str, required=True, help="Name of the ChromaDB collection")

    args = parser.parse_args()
    file_path = args.jsonl
    items = []
    # uncomment this to store embeddings in ChromaDB
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    
    for item in tqdm(load_jsonl(file_path), total=total_lines, desc="Processing items"):
        formatted_text = format_item_text(item)
        embedding = get_ollama_embedding(formatted_text)  # OR use get_mixedbread_embedding
        
        items.append({
            'id': item['id'],
            'name': item.get('name', ''),
            'description': item.get('description', ''),
            'style': item.get('style', ''),
            'category': item.get('category', ''),
            'buy_url': item.get('buy_url', ''),  # Handle missing URLs gracefully
            'embedding': embedding
        })
    
    # Store in FAISS or ChromaDB
    store_in_chromadb(items)
    
    print("Embeddings stored successfully!")
    """

    # Example search
    query = "Puff-sleeve open-back faux-leather top"
    results = search_chromadb(query)
    print("Search results:", results)

if __name__ == "__main__":
    main()
