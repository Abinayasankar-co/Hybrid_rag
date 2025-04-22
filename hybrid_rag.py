import pymongo
from sentence_transformers import SentenceTransformer
import numpy as np

# Connect to MongoDB
mongo_uri = "mongodb://localhost:27017/"  # Default local URI
client = pymongo.MongoClient(mongo_uri)
db = client["rag_database"]
collection = db["documents"]

print("Connected to MongoDB successfully!")

# Sample document chunks
documents = [
    "The Battle of Waterloo was fought in 1815.",
    "Napoleon Bonaparte led the French army.",
    "The battle took place near Brussels.",
    "Wellington commanded the allied forces.",
    "The French army was defeated in the battle."
]

# Create embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedding_model.encode(documents)
embeddings = np.array(embeddings).astype('float32')

# Store data in MongoDB
collection.delete_many({})  # Clear existing data
for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
    collection.insert_one({"chunk_id": i, "text": doc, "embedding": embedding.tolist()})

print("Data stored in MongoDB!")
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer

# Create FAISS index for vector search
dimension = embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(embeddings)

# Create TF-IDF index for keyword search
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

print("Search indexes created!")
def hybrid_retrieval(query, top_k=2, alpha=0.5):
    # Vector search
    query_embedding = embedding_model.encode([query]).astype('float32')
    D, I = faiss_index.search(query_embedding, top_k)
    vector_results = [(documents[i], D[0][j]) for j, i in enumerate(I[0])]

    # Keyword search
    query_tfidf = vectorizer.transform([query])
    scores = (tfidf_matrix @ query_tfidf.T).toarray().flatten()
    keyword_indices = scores.argsort()[-top_k:][::-1]
    keyword_results = [(documents[i], scores[i]) for i in keyword_indices]

    # Combine results
    combined_results = {}
    for chunk, score in vector_results:
        combined_results[chunk] = alpha * score
    for chunk, score in keyword_results:
        if chunk in combined_results:
            combined_results[chunk] += (1 - alpha) * score
        else:
            combined_results[chunk] = (1 - alpha) * score

    # Sort and return top results
    sorted_results = sorted(combined_results.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [result[0] for result in sorted_results]

# Test the retrieval
query = "Who led the French army at Waterloo?"
retrieved_chunks = hybrid_retrieval(query)
print("Retrieved Chunks:", retrieved_chunks)
import requests

def generate_response_with_grok(query, context):
    grok_api_url = "https://api.groq.com/openai/v1/chat/completions"  # Correct URL
    api_key = "xai-ntUPrmWHBL2I2Mqbk8F7cQPEwhd6CIg32NyYR5osyMbRfF36yBIeliCH0j5Y7zGmdSLzlRR4OIffe14E"  # Replace with your valid API key

    # Prepare the prompt as a message
    prompt = f"Question: {query}\nContext: {context}\nAnswer:"
    messages = [{"role": "user", "content": prompt}]  # Format expected by Grok API

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "grok-3",  # Use grok-3 (released March 10, 2025) or check available models
        "messages": messages,
        "max_tokens": 100,
        "temperature": 0.7,
        "stream": False  # Optional, to get a single response
    }

    try:
        response = requests.post(grok_api_url, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        # Adjust based on actual response structure (e.g., choices[0].message.content)
        return result.get("choices", [{}])[0].get("message", {}).get("content", "Error: No response from Grok API")
    except requests.exceptions.RequestException as e:
        return f"Error calling Grok API: {str(e)}"
    import pymongo
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import requests

# Connect to MongoDB
mongo_uri = "mongodb://localhost:27017/"
client = pymongo.MongoClient(mongo_uri)
db = client["rag_database"]
collection = db["documents"]

# Prepare and store data
documents = [
    "The Battle of Waterloo was fought in 1815.",
    "Napoleon Bonaparte led the French army.",
    "The battle took place near Brussels.",
    "Wellington commanded the allied forces.",
    "The French army was defeated in the battle."
]
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedding_model.encode(documents)
embeddings = np.array(embeddings).astype('float32')
collection.delete_many({})
for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
    collection.insert_one({"chunk_id": i, "text": doc, "embedding": embedding.tolist()})

# Set up search indexes
dimension = embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(embeddings)
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# Hybrid retrieval function
def hybrid_retrieval(query, top_k=2, alpha=0.5):
    query_embedding = embedding_model.encode([query]).astype('float32')
    D, I = faiss_index.search(query_embedding, top_k)
    vector_results = [(documents[i], D[0][j]) for j, i in enumerate(I[0])]

    query_tfidf = vectorizer.transform([query])
    scores = (tfidf_matrix @ query_tfidf.T).toarray().flatten()
    keyword_indices = scores.argsort()[-top_k:][::-1]
    keyword_results = [(documents[i], scores[i]) for i in keyword_indices]

    combined_results = {}
    for chunk, score in vector_results:
        combined_results[chunk] = alpha * score
    for chunk, score in keyword_results:
        if chunk in combined_results:
            combined_results[chunk] += (1 - alpha) * score
        else:
            combined_results[chunk] = (1 - alpha) * score

    sorted_results = sorted(combined_results.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [result[0] for result in sorted_results]

import requests

def generate_response_with_grok(query, context):
    groq_api_url = "https://api.groq.com/openai/v1/chat/completions"  # Groq API endpoint
    api_key = "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"  # Replace with your Groq API key

    prompt = f"Question: {query}\nContext: {context}\nAnswer:"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.3-70b-versatile",  # Example Groq model (check console.groq.com for options)
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 100,
        "temperature": 0.7
    }

    try:
        response = requests.post(groq_api_url, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        print("API Response:", result)  # Debug: Print the response
        return result["choices"][0]["message"]["content"]  # Groq returns content here
    except requests.exceptions.RequestException as e:
        print("Error Details:", e.response.text if e.response else str(e))
        return f"Error calling Groq API: {str(e)}"
# Main function to run the system
if __name__ == "__main__":
    query = "Who led the French army at Waterloo?"
    retrieved_chunks = hybrid_retrieval(query)
    context = " ".join(retrieved_chunks)
    response = generate_response_with_grok(query, context)

    print("Query:", query)
    print("Retrieved Chunks:", retrieved_chunks)
    print("Generated Response:", response)

    client.close()  # Close MongoDB connection