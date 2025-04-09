import ollama
import numpy as np
import os
import time
from typing import List, Dict, Tuple
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import pandas as pd
import json
from datetime import datetime

load_dotenv()

# Embedding model used
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest")
OLLAMA_NUM_PARALLEL = os.getenv("OLLAMA_NUM_PARALLEL", "1")

def chunk_text(text: str, chunk_size: int, max_characters: int) -> List[str]:
    """Split text into chunks of specified size"""
    chunks = []
    for i in range(0, len(text) if len(text) < max_characters else max_characters, chunk_size):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)
    return chunks

def load_sample_text(filename: str = "16-h.htm") -> str:
    """Load sample text from file"""
    print(f"Loading text from {filename}...")
    try:
        with open(filename) as fd:
            return fd.read()
    except FileNotFoundError:
        print(f"Warning: {filename} not found. Using sample text instead.")
        return "This is a sample text. " * 100

def embed_string(s: str) -> np.ndarray:
    """Embed a single string"""
    return np.array(ollama.embed(
        input=s,
        model=EMBEDDING_MODEL,
        options={},
        truncate=False
    )["embeddings"])[0]

def embed_list(s: List[str]) -> np.ndarray:
    """Embed a list of strings in a single batch"""
    return np.array(ollama.embed(
        input=s,
        model=EMBEDDING_MODEL,
        options={},
        truncate=False
    )["embeddings"])

def measure_performance(chunks: List[str], batch_sizes: List[int], num_trials: int = 3) -> Dict:
    """
    Measure embedding performance for different batch sizes
    
    Args:
        chunks: List of text chunks to embed
        batch_sizes: List of batch sizes to test
        num_trials: Number of trials to run for each batch size
        
    Returns:
        Dictionary with performance metrics
    """
    results = {
        "batch_sizes": batch_sizes,
        "individual_times": [],       # Time for embedding individually
        "batch_times": [],            # Time for embedding in batches
        "speedup_ratios": [],         # Batch speed / Individual speed
        "chunks_per_second_individual": [],
        "chunks_per_second_batch": [],
        "metadata": {
            "model": EMBEDDING_MODEL,
            "ollama_num_parallel": OLLAMA_NUM_PARALLEL,
            "num_trials": num_trials,
            "timestamp": datetime.now().isoformat()
        }
    }
    
    for batch_size in batch_sizes:
        print(f"\nTesting batch size: {batch_size}")
        
        # Limit to the current batch size for testing
        test_chunks = chunks[:batch_size]
        num_chunks = len(test_chunks)
        
        # Individual embedding timing (multiple trials)
        individual_times = []
        for trial in range(num_trials):
            print(f"  Individual embedding trial {trial+1}/{num_trials}...")
            start_time = time.time()
            # Process each chunk individually
            for chunk in test_chunks:
                _ = embed_string(chunk)
            end_time = time.time()
            trial_time = end_time - start_time
            individual_times.append(trial_time)
            print(f"    Time: {trial_time:.2f}s, Rate: {num_chunks/trial_time:.2f} chunks/sec")
        
        avg_individual_time = np.mean(individual_times)
        results["individual_times"].append(avg_individual_time)
        results["chunks_per_second_individual"].append(num_chunks / avg_individual_time)
        
        # Batch embedding timing (multiple trials)
        batch_times = []
        for trial in range(num_trials):
            print(f"  Batch embedding trial {trial+1}/{num_trials}...")
            start_time = time.time()
            # Process all chunks in one batch
            _ = embed_list(test_chunks)
            end_time = time.time()
            trial_time = end_time - start_time
            batch_times.append(trial_time)
            print(f"    Time: {trial_time:.2f}s, Rate: {num_chunks/trial_time:.2f} chunks/sec")
        
        avg_batch_time = np.mean(batch_times)
        results["batch_times"].append(avg_batch_time)
        results["chunks_per_second_batch"].append(num_chunks / avg_batch_time)
        
        # Calculate speedup
        speedup = avg_individual_time / avg_batch_time
        results["speedup_ratios"].append(speedup)
        print(f"  Average speedup: {speedup:.2f}x")
    
    return results

def plot_performance_results(results: Dict):
    """Create visualizations for performance results"""
    batch_sizes = results["batch_sizes"]
    batch_sizes_str = [str(size) for size in batch_sizes]
    
    plt.figure(figsize=(16, 12))
    
    # Plot 1: Execution Time Comparison
    plt.subplot(2, 2, 1)
    width = 0.35
    x = np.arange(len(batch_sizes))
    plt.bar(x - width/2, results["individual_times"], width, label='Individual')
    plt.bar(x + width/2, results["batch_times"], width, label='Batch')
    plt.xlabel('Batch Size')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Execution Time Comparison')
    plt.xticks(x, batch_sizes_str)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 2: Speedup Ratio
    plt.subplot(2, 2, 2)
    plt.plot(batch_sizes_str, results["speedup_ratios"], 'o-', color='green', linewidth=2, markersize=8)
    plt.axhline(y=1, color='r', linestyle='--', alpha=0.7)
    plt.xlabel('Batch Size')
    plt.ylabel('Speedup Ratio (Individual/Batch)')
    plt.title('Batch Processing Speedup')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 3: Throughput Comparison
    plt.subplot(2, 2, 3)
    plt.plot(batch_sizes_str, results["chunks_per_second_individual"], 'o-', label='Individual', color='blue', linewidth=2, markersize=8)
    plt.plot(batch_sizes_str, results["chunks_per_second_batch"], 'o-', label='Batch', color='orange', linewidth=2, markersize=8)
    plt.xlabel('Batch Size')
    plt.ylabel('Chunks Per Second')
    plt.title('Throughput Comparison')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 4: Normalized Performance
    plt.subplot(2, 2, 4)
    base_individual = results["chunks_per_second_individual"][0]  # For batch size 1
    base_batch = results["chunks_per_second_batch"][0]  # For batch size 1
    
    norm_individual = [x / base_individual for x in results["chunks_per_second_individual"]]
    norm_batch = [x / base_batch for x in results["chunks_per_second_batch"]]
    
    plt.plot(batch_sizes_str, norm_individual, 'o-', label='Individual', color='blue', linewidth=2, markersize=8)
    plt.plot(batch_sizes_str, norm_batch, 'o-', label='Batch', color='orange', linewidth=2, markersize=8)
    plt.xlabel('Batch Size')
    plt.ylabel('Normalized Throughput')
    plt.title('Scaling Efficiency')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add metadata as text
    plt.figtext(0.5, 0.01, 
                f"Model: {results['metadata']['model']} | OLLAMA_NUM_PARALLEL: {results['metadata']['ollama_num_parallel']} | Trials: {results['metadata']['num_trials']}", 
                ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ollama_performance_{timestamp}.png"
    plt.savefig(filename, dpi=300)
    print(f"Performance visualization saved as {filename}")
    plt.show()
    
    # Also save the raw results
    with open(f"ollama_performance_{timestamp}.json", 'w') as f:
        json.dump(results, f, indent=2)

def main():
    print(f"Starting Ollama batch performance test with model: {EMBEDDING_MODEL}")
    print(f"OLLAMA_NUM_PARALLEL setting: {OLLAMA_NUM_PARALLEL}")
    
    # Load test text
    text = load_sample_text()
    
    # Create chunks
    chunk_size = 256  # Size of each text chunk
    max_chars = 100000  # Limit text processing to avoid too long tests
    chunks = chunk_text(text, chunk_size, max_chars)
    print(f"Created {len(chunks)} chunks of text")
    
    # Define batch sizes to test
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    
    # Measure performance
    results = measure_performance(chunks, batch_sizes, num_trials=3)
    
    # Plot results
    plot_performance_results(results)
    
    # Print summary
    print("\nPERFORMANCE TEST SUMMARY:")
    print("-" * 50)
    print(f"{'Batch Size':<10} {'Individual (s)':<15} {'Batch (s)':<15} {'Speedup':<10}")
    print("-" * 50)
    for i, batch_size in enumerate(results["batch_sizes"]):
        print(f"{batch_size:<10} {results['individual_times'][i]:<15.2f} {results['batch_times'][i]:<15.2f} {results['speedup_ratios'][i]:<10.2f}x")

if __name__ == "__main__":
    main()