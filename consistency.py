import ollama
import numpy as np
import os
from typing import List
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

load_dotenv()

# Embedding model used was "nomic-embed-text"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
EPS=1e-4

def chunk_text(text: str, chunk_size: int, max_characters: int) -> List[str]:
    chunks = []
    for i in range(0, len(text) if len(text) < max_characters else max_characters, chunk_size):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)
    return chunks

# Used first few chapters of Peter Pan
text = ""
with open("16-h.htm") as fd:
    text = fd.read()

chunk_size = 256
# 256 is the max batch size that is defined later
chunks = chunk_text(text, chunk_size, chunk_size * 256)

def embed_string(s: str) -> np.ndarray:
    return np.array(ollama.embed(
        input=s,
        model=EMBEDDING_MODEL,
        options={
            
        },
        truncate=False
    )["embeddings"])[0]

def embed_list(s: List[str]) -> np.ndarray:
    return np.array(ollama.embed(
        input=s,
        model=EMBEDDING_MODEL,
        options={
            
        },
        truncate=False
    )["embeddings"])

def test(list_of_string: List[str], batch_sizes: List[int]) -> tuple:
    avg_distances = []
    avg_similarites = []

    max_distances = []
    min_similarities = []

    for batch_size in batch_sizes:
        print(f"Results for batch size: {batch_size}")
        singles = np.array([embed_string(s) for s in list_of_string[:batch_size]])
        as_list = embed_list(list_of_string[:batch_size])
    
        distances = []
        for single_embedding, as_list_embedding in zip(singles, as_list):
            distance = np.sqrt(((single_embedding - as_list_embedding) ** 2).sum())
            distances.append(distance)

        distances = np.array(distances)

        mean = np.mean(distances)
        max_val = np.max(distances)

        avg_distances.append(mean)
        max_distances.append(max_val)

        print("Euclidean Distance:")
        print(f"\tMean of euclidean distances: {mean}")
        print(f"\tMax euclidean distance: {max_val}")
        
        # Cosine similarity
        similarities = []
        for single_embedding, as_list_embedding in zip(singles, as_list):
            vector1 = single_embedding.reshape(1, -1)
            vector2 = as_list_embedding.reshape(1, -1)
            similarity = cosine_similarity(vector1, vector2)
            similarities.append(similarity)


        similarities = np.array(similarities)

        mean = np.mean(similarities)  
        min_val = np.min(similarities)

        avg_similarites.append(mean)
        min_similarities.append(min_val)

        print("Cosine Similarity:")
        print(f"\tMean of cosine similarites: {mean}")
        print(f"\tMin cosine similarity: {min_val}")

        print("==========================================================")

    return (batch_sizes, avg_distances, avg_similarites, max_distances, min_similarities)

def plot_results(batch_sizes, avg_distances, avg_similarities, max_distances, min_similarities):
    """
    Plot the results from the embedding test in a 2x2 grid
    """
    # Convert batch sizes to strings for better x-axis labeling
    batch_size_labels = [str(size) for size in batch_sizes]
    
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Average Euclidean Distance vs Batch Size
    plt.subplot(2, 2, 1)
    plt.plot(batch_size_labels, avg_distances, 'o-', color='blue', linewidth=2, markersize=8)
    plt.title('Average Euclidean Distance vs Batch Size', fontsize=14)
    plt.xlabel('Batch Size', fontsize=12)
    plt.ylabel('Average Distance', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    
    # Plot 2: Max Euclidean Distance vs Batch Size
    plt.subplot(2, 2, 2)
    plt.plot(batch_size_labels, max_distances, 'o-', color='red', linewidth=2, markersize=8)
    plt.title('Maximum Euclidean Distance vs Batch Size', fontsize=14)
    plt.xlabel('Batch Size', fontsize=12)
    plt.ylabel('Max Distance', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    
    # Plot 3: Average Cosine Similarity vs Batch Size
    plt.subplot(2, 2, 3)
    plt.plot(batch_size_labels, avg_similarities, 'o-', color='green', linewidth=2, markersize=8)
    plt.title('Average Cosine Similarity vs Batch Size', fontsize=14)
    plt.xlabel('Batch Size', fontsize=12)
    plt.ylabel('Average Similarity', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    
    # Plot 4: Min Cosine Similarity vs Batch Size
    plt.subplot(2, 2, 4)
    plt.plot(batch_size_labels, min_similarities, 'o-', color='purple', linewidth=2, markersize=8)
    plt.title('Minimum Cosine Similarity vs Batch Size', fontsize=14)
    plt.xlabel('Batch Size', fontsize=12)
    plt.ylabel('Min Similarity', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('embedding_batch_performance.png', dpi=300)
    print("Visualization saved as 'embedding_batch_performance.png'")
    plt.show()

# Run the test
batch_sizes_list = [2**i for i in range(1, 12)]
batch_sizes, avg_distances, avg_similarities, max_distances, min_similarities = test(chunks, batch_sizes_list)

# Visualize the results
plot_results(batch_sizes, avg_distances, avg_similarities, max_distances, min_similarities)

# Print summary of findings
print("\n=== SUMMARY OF FINDINGS ===")
print(f"Best batch size for minimizing Euclidean distance: {batch_sizes[np.argmin(avg_distances)]}")
print(f"Best batch size for maximizing Cosine similarity: {batch_sizes[np.argmax(avg_similarities)]}")
print("\nRecommended batch size based on overall performance: ", end="")
# Find a reasonable recommendation based on both metrics
euclidean_rank = np.argsort(avg_distances)
cosine_rank = np.argsort(-np.array(avg_similarities))  # Negative to sort in descending order
combined_rank = euclidean_rank + cosine_rank
recommended_batch_size = batch_sizes[np.argmin(combined_rank)]
print(f"{recommended_batch_size}")