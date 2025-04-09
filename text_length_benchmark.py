import ollama

def analyze_results(results):
    """Generate a text summary of the benchmark findings"""
    text_lengths = results["text_lengths"]
    
    # Find optimal batch sizes for each text length
    optimal_settings = {}
    
    for length in text_lengths:
        if length not in results["performance"] or len(results["performance"][length]["speedup_ratios"]) == 0:
            continue
            
        perf = results["performance"][length]
        batch_sizes = results["batch_sizes"][:len(perf["speedup_ratios"])]
        
        # Find batch size with best speedup
        best_speedup_idx = np.argmax(perf["speedup_ratios"])
        best_speedup = perf["speedup_ratios"][best_speedup_idx]
        best_speedup_size = batch_sizes[best_speedup_idx]
        
        # Find batch size with best throughput
        best_throughput_idx = np.argmax(perf["chunks_per_second_batch"])
        best_throughput = perf["chunks_per_second_batch"][best_throughput_idx]
        best_throughput_size = batch_sizes[best_throughput_idx]
        
        # Store results
        optimal_settings[length] = {
            "best_speedup": {
                "batch_size": best_speedup_size,
                "value": best_speedup
            },
            "best_throughput": {
                "batch_size": best_throughput_size,
                "value": best_throughput,
                "latency": perf["batch_times"][best_throughput_idx]
            }
        }
    
    # Generate summary text
    summary = [
        "# OLLAMA EMBEDDING BENCHMARK SUMMARY",
        f"Model: {results['model']} | OLLAMA_NUM_PARALLEL: {results['ollama_num_parallel']}",
        "",
        "## Optimal Batch Sizes by Text Length",
        ""
    ]
    
    summary.append("| Text Length | Best Speedup | Best Throughput | Latency (ms) |")
    summary.append("|------------|-------------|----------------|--------------|")
    
    for length in text_lengths:
        if length not in optimal_settings:
            continue
            
        settings = optimal_settings[length]
        speedup_info = f"{settings['best_speedup']['batch_size']} ({settings['best_speedup']['value']:.2f}x)"
        throughput_info = f"{settings['best_throughput']['batch_size']} ({settings['best_throughput']['value']:.2f} chunks/s)"
        latency_ms = f"{settings['best_throughput']['latency'] * 1000:.1f}"
        
        summary.append(f"| {length} chars | {speedup_info} | {throughput_info} | {latency_ms} |")
    
    summary.append("")
    summary.append("## Key Findings")
    summary.append("")
    
    # Analyze throughput scaling with batch size
    throughput_scaling = {}
    for length in text_lengths:
        if length not in results["performance"] or len(results["performance"][length]["chunks_per_second_batch"]) < 2:
            continue
            
        throughput = results["performance"][length]["chunks_per_second_batch"]
        scaling = [throughput[i+1]/throughput[i] for i in range(len(throughput)-1)]
        throughput_scaling[length] = scaling
    
    # Check for diminishing returns
    diminishing_returns = {}
    for length, scaling in throughput_scaling.items():
        batch_sizes = results["batch_sizes"][:len(scaling)+1]
        for i, scale_factor in enumerate(scaling):
            if scale_factor < 1.1:  # Less than 10% improvement
                diminishing_returns[length] = batch_sizes[i+1]
                break
    
    # Add findings
    summary.append("1. **Batch Size Impact by Text Length:**")
    for length in text_lengths:
        if length not in optimal_settings:
            continue
            
        settings = optimal_settings[length]
        summary.append(f"   - {length} chars: Optimal batch size {settings['best_throughput']['batch_size']} for throughput, {settings['best_speedup']['batch_size']} for efficiency")
    
    summary.append("")
    summary.append("2. **Diminishing Returns Threshold:**")
    for length, batch_size in diminishing_returns.items():
        summary.append(f"   - {length} chars: Diminishing returns observed beyond batch size {batch_size}")
    
    summary.append("")
    summary.append("3. **Latency Considerations:**")
    for length in text_lengths:
        if length not in optimal_settings:
            continue
            
        settings = optimal_settings[length]
        summary.append(f"   - {length} chars: Batch size {settings['best_throughput']['batch_size']} has {settings['best_throughput']['latency']:.2f}s latency")
    
    summary.append("")
    summary.append("## Recommendations for BitIQ Nostr_AI Service")
    summary.append("")
    summary.append("### Short User Queries (15-100 chars):")
    
    # Generate specific recommendations for BitIQ use cases
    short_query_batch = None
    for length in [l for l in text_lengths if l <= 100]:
        if length in optimal_settings:
            short_query_batch = optimal_settings[length]['best_speedup']['batch_size']
            break
    
    if short_query_batch:
        summary.append(f"- Use batch size {short_query_batch} for optimal processing of user search queries")
        summary.append("- Consider dedicated processing path with minimal wait times (20-50ms)")
        
    summary.append("")
    summary.append("### Discussion Content (256+ chars):")
    
    long_content_batch = None
    for length in [l for l in text_lengths if l >= 256]:
        if length in optimal_settings:
            long_content_batch = optimal_settings[length]['best_throughput']['batch_size']
            break
    
    if long_content_batch:
        summary.append(f"- Use batch size {long_content_batch} for processing longer discussion content")
        summary.append("- Can use longer wait times to build batches (100-300ms)")
    
    return "\n".join(summary)

def visualize_results(results):
    """Create comprehensive visualizations of the benchmark results"""
    text_lengths = results["text_lengths"]
    batch_sizes = results["batch_sizes"]
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f'Ollama Embedding Performance by Text Length\n'
                f'Model: {results["model"]}, OLLAMA_NUM_PARALLEL: {results["ollama_num_parallel"]}', 
                fontsize=16)
    
    # Plot layout
    gs = fig.add_gridspec(3, 2)  # 3 rows, 2 columns
    
    # Create DataFrames for easier plotting
    perf_data = {}
    
    for length in text_lengths:
        if length not in results["performance"]:
            continue
            
        # Add performance metrics to DataFrame
        perf_data[str(length)] = {}
        
        # Get the actual measurements for this text length
        speedup_ratios = results["performance"][length]["speedup_ratios"]
        throughput = results["performance"][length]["chunks_per_second_batch"]
        
        # Get the batch sizes that were actually measured for this text length
        measured_batch_sizes = batch_sizes[:len(speedup_ratios)]
        measured_batch_sizes_str = [str(size) for size in measured_batch_sizes]
        
        perf_data[str(length)]['batch_sizes'] = measured_batch_sizes
        perf_data[str(length)]['batch_sizes_str'] = measured_batch_sizes_str
        perf_data[str(length)]['speedup'] = speedup_ratios
        perf_data[str(length)]['throughput'] = throughput
    
    # 1. Speedup Ratio by Text Length
    ax1 = fig.add_subplot(gs[0, 0])
    for length in text_lengths:
        length_str = str(length)
        if length_str in perf_data and len(perf_data[length_str]['speedup']) > 0:
            ax1.plot(perf_data[length_str]['batch_sizes_str'], 
                     perf_data[length_str]['speedup'], 
                     'o-', label=f'{length} chars')
    
    ax1.axhline(y=1, color='r', linestyle='--', alpha=0.7)
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Speedup Ratio (Individual/Batch)')
    ax1.set_title('Batch Processing Speedup by Text Length')
    ax1.legend(title='Text Length')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 2. Throughput by Text Length
    ax2 = fig.add_subplot(gs[0, 1])
    for length in text_lengths:
        length_str = str(length)
        if length_str in perf_data and len(perf_data[length_str]['throughput']) > 0:
            ax2.plot(perf_data[length_str]['batch_sizes_str'], 
                     perf_data[length_str]['throughput'], 
                     'o-', label=f'{length} chars')
    
    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('Chunks Per Second')
    ax2.set_title('Batch Throughput by Text Length')
    ax2.legend(title='Text Length')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # 3. Heatmap for Speedup Ratio
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Find the maximum number of batch sizes actually measured for any text length
    max_measured_batch_sizes = 0
    for length in text_lengths:
        if length in results["performance"]:
            max_measured_batch_sizes = max(max_measured_batch_sizes, 
                                       len(results["performance"][length]["speedup_ratios"]))
    
    # Use only batch sizes that were measured
    measured_batch_sizes = batch_sizes[:max_measured_batch_sizes]
    
    speedup_data = np.zeros((len(text_lengths), len(measured_batch_sizes)))
    speedup_mask = np.ones(speedup_data.shape, dtype=bool)
    
    for i, length in enumerate(text_lengths):
        if length in results["performance"]:
            ratios = results["performance"][length]["speedup_ratios"]
            for j in range(len(ratios)):
                speedup_data[i, j] = ratios[j]
                speedup_mask[i, j] = False
    
    # Create masked array for proper heatmap display
    masked_speedup = np.ma.array(speedup_data, mask=speedup_mask)
    
    im = ax3.imshow(masked_speedup, cmap='viridis')
    ax3.set_xticks(np.arange(len(measured_batch_sizes)))
    ax3.set_yticks(np.arange(len(text_lengths)))
    ax3.set_xticklabels(measured_batch_sizes)
    ax3.set_yticklabels(text_lengths)
    ax3.set_xlabel('Batch Size')
    ax3.set_ylabel('Text Length (chars)')
    ax3.set_title('Speedup Ratio Heatmap')
    
    # Add text annotations for non-masked values
    for i in range(len(text_lengths)):
        for j in range(len(measured_batch_sizes)):
            if not speedup_mask[i, j]:
                text = ax3.text(j, i, f"{speedup_data[i, j]:.1f}",
                              ha="center", va="center", color="w" if speedup_data[i, j] < 2 else "black")
    
    plt.colorbar(im, ax=ax3)
    
    # 4. Heatmap for Throughput
    ax4 = fig.add_subplot(gs[1, 1])
    
    throughput_data = np.zeros((len(text_lengths), len(measured_batch_sizes)))
    throughput_mask = np.ones(throughput_data.shape, dtype=bool)
    
    for i, length in enumerate(text_lengths):
        if length in results["performance"]:
            throughputs = results["performance"][length]["chunks_per_second_batch"]
            for j in range(len(throughputs)):
                throughput_data[i, j] = throughputs[j]
                throughput_mask[i, j] = False
    
    # Create masked array for proper heatmap display
    masked_throughput = np.ma.array(throughput_data, mask=throughput_mask)
    
    im = ax4.imshow(masked_throughput, cmap='plasma')
    ax4.set_xticks(np.arange(len(measured_batch_sizes)))
    ax4.set_yticks(np.arange(len(text_lengths)))
    ax4.set_xticklabels(measured_batch_sizes)
    ax4.set_yticklabels(text_lengths)
    ax4.set_xlabel('Batch Size')
    ax4.set_ylabel('Text Length (chars)')
    ax4.set_title('Throughput Heatmap (chunks/second)')
    
    # Calculate max throughput of non-masked values for color threshold
    max_visible_throughput = np.max(throughput_data[~throughput_mask]) if np.any(~throughput_mask) else 1
    
    # Add text annotations for non-masked values
    for i in range(len(text_lengths)):
        for j in range(len(measured_batch_sizes)):
            if not throughput_mask[i, j]:
                text = ax4.text(j, i, f"{throughput_data[i, j]:.1f}",
                              ha="center", va="center", 
                              color="w" if throughput_data[i, j] < max_visible_throughput/2 else "black")
    
    plt.colorbar(im, ax=ax4)
    
    # 5. Batch Size Recommendations
    ax5 = fig.add_subplot(gs[2, 0])
    
    # Calculate optimal batch sizes
    optimal_speedup = {}
    optimal_throughput = {}
    
    for length in text_lengths:
        # Find batch size with highest speedup
        if length in results["performance"] and len(results["performance"][length]["speedup_ratios"]) > 0:
            speedup_values = results["performance"][length]["speedup_ratios"]
            best_speedup_idx = np.argmax(speedup_values)
            optimal_speedup[length] = batch_sizes[:len(speedup_values)][best_speedup_idx]
            
            # Find batch size with highest throughput
            throughput_values = results["performance"][length]["chunks_per_second_batch"]
            best_throughput_idx = np.argmax(throughput_values)
            optimal_throughput[length] = batch_sizes[:len(throughput_values)][best_throughput_idx]
    
    # Prepare data for bar chart
    bar_data = []
    bar_labels = []
    
    for length in text_lengths:
        if length in optimal_speedup and length in optimal_throughput:
            bar_data.append([optimal_speedup[length], optimal_throughput[length]])
            bar_labels.append(str(length))
    
    if bar_data:
        bar_data = np.array(bar_data).T
        width = 0.35
        x = np.arange(len(bar_labels))
        
        ax5.bar(x - width/2, bar_data[0], width, label='Best Speedup')
        ax5.bar(x + width/2, bar_data[1], width, label='Best Throughput')
        
        ax5.set_xlabel('Text Length (chars)')
        ax5.set_ylabel('Optimal Batch Size')
        ax5.set_title('Recommended Batch Sizes by Optimization Goal')
        ax5.set_xticks(x)
        ax5.set_xticklabels(bar_labels)
        ax5.legend()
    else:
        ax5.text(0.5, 0.5, "Insufficient data for recommendations", 
                ha='center', va='center', transform=ax5.transAxes)
    
    # 6. Batch Size vs Latency
    ax6 = fig.add_subplot(gs[2, 1])
    
    for length in text_lengths:
        length_str = str(length)
        if length_str in perf_data and 'batch_sizes_str' in perf_data[length_str]:
            batch_times = results["performance"][length]["batch_times"]
            if len(batch_times) > 0:
                ax6.plot(perf_data[length_str]['batch_sizes_str'], 
                        batch_times, 
                        'o-', label=f'{length} chars')
    
    ax6.set_xlabel('Batch Size')
    ax6.set_ylabel('Processing Time (seconds)')
    ax6.set_title('Batch Processing Latency by Text Length')
    ax6.legend(title='Text Length')
    ax6.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"ollama_{results['model']}_benchmark_{timestamp}.png"
    plt.savefig(plot_filename, dpi=300)
    print(f"\nVisualization saved as {plot_filename}")
    
    return plot_filename
import numpy as np
import os
import time
from typing import List, Dict, Tuple
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import pandas as pd
import json
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

# Embedding model used
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest")
OLLAMA_NUM_PARALLEL = os.getenv("OLLAMA_NUM_PARALLEL", "1")

print(f"Using embedding model: {EMBEDDING_MODEL}")
print(f"OLLAMA_NUM_PARALLEL setting: {OLLAMA_NUM_PARALLEL}")

def load_sample_text(filename: str = "16-h.htm") -> str:
    """Load sample text from file"""
    print(f"Loading text from {filename}...")
    try:
        with open(filename) as fd:
            return fd.read()
    except FileNotFoundError:
        print(f"Warning: {filename} not found. Using sample text instead.")
        # Generate a more diverse sample text with varying sentence structures
        sample = []
        for i in range(50):
            if i % 5 == 0:
                sample.append(f"Short query {i}.")
            elif i % 5 == 1:
                sample.append(f"Medium length sentence about topic {i} with some additional context.")
            elif i % 5 == 2:
                sample.append(f"This is a longer paragraph that discusses the details of topic {i} in more depth. " 
                            f"It contains multiple sentences and explores various aspects of the subject matter.")
            elif i % 5 == 3:
                sample.append(f"Technical description {i}: This text includes specific terminology and structured "
                            f"information that might be found in a technical document or specification. "
                            f"It contains precise details about implementation, configuration settings, "
                            f"and references to related concepts.")
            else:
                sample.append(f"Conversational thread {i}: This represents a back-and-forth discussion "
                            f"between multiple participants. It includes questions, responses, "
                            f"disagreements, clarifications, and references to earlier points "
                            f"in the conversation. The language is more informal and includes "
                            f"various viewpoints on the topic.")
        return " ".join(sample)

def chunk_text_fixed_length(text: str, target_length: int, max_chunks: int = 100) -> List[str]:
    """
    Create chunks of approximately the target length from the text.
    
    Args:
        text: The source text to chunk
        target_length: Target length in characters for each chunk
        max_chunks: Maximum number of chunks to create
        
    Returns:
        List of text chunks with approximately target_length characters
    """
    chunks = []
    
    # For very short targets (like search queries), generate realistic queries
    if target_length <= 30:
        search_queries = [
            "bitcoin mining",
            "nostr protocol implementation",
            "decentralized social media",
            "vector similarity search",
            "golang error handling best practices",
            "embedding models comparison",
            "site reliability engineering",
            "distributed systems consensus",
            "bitcoin lightning network",
            "web of trust implementation",
            "zero knowledge proofs",
            "blockchain privacy",
            "Bitcoin transaction fees",
            "Ollama batch processing",
            "microservices architecture",
            "kubernetes deployment strategies",
            "AI vector embeddings",
            "decentralized identity",
            "data synchronization techniques",
            "openshift gitops"
        ]
        
        # Add variations of similar length to reach max_chunks
        variations = []
        for query in search_queries:
            if len(variations) >= max_chunks:
                break
            variations.append(query)
            if len(variations) >= max_chunks:
                break
            variations.append(f"how to {query}")
            if len(variations) >= max_chunks:
                break
            variations.append(f"best {query} tutorial")
            if len(variations) >= max_chunks:
                break
            variations.append(f"{query} vs alternatives")
            if len(variations) >= max_chunks:
                break
            variations.append(f"optimizing {query}")
        
        return variations[:max_chunks]
    
    # For medium/long content, chunk the source text
    words = text.split()
    current_chunk = []
    current_length = 0
    
    for word in words:
        word_length = len(word) + 1  # +1 for the space
        
        if current_length + word_length > target_length and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = word_length
        else:
            current_chunk.append(word)
            current_length += word_length
        
        if len(chunks) >= max_chunks:
            break
    
    # Add the last chunk if it exists and we haven't reached max_chunks
    if current_chunk and len(chunks) < max_chunks:
        chunks.append(" ".join(current_chunk))
    
    return chunks

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

def test_length_impact(text_lengths=None, batch_sizes=None, num_trials=2, model_name=None):
    """
    Test impact of different text lengths on embedding performance
    
    Args:
        text_lengths: List of text lengths to test
        batch_sizes: List of batch sizes to test
        num_trials: Number of trials for each configuration
        model_name: Name of model being tested (for logging)
        
    Returns:
        Dictionary with performance metrics
    """
    if text_lengths is None:
        text_lengths = [15, 100, 256, 480]  # Characters
    
    if batch_sizes is None:
        batch_sizes = [1, 2, 8, 16, 32, 48, 64, 80]
    
    # Load sample text
    text = load_sample_text()
    
    results = {
        "model": EMBEDDING_MODEL,
        "ollama_num_parallel": OLLAMA_NUM_PARALLEL,
        "text_lengths": text_lengths,
        "batch_sizes": batch_sizes,
        "performance": {},
        "consistency": {},
        "timestamp": datetime.now().isoformat()
    }
    
    for length in text_lengths:
        print(f"\n{'-'*60}")
        print(f"Testing text length: {length} characters")
        print(f"{'-'*60}")
        
        # Create chunks with specific length
        chunks = chunk_text_fixed_length(text, length, max_chunks=max(batch_sizes)*2)
        print(f"Created {len(chunks)} chunks of ~{length} characters each")
        
        # Initialize results for this length
        results["performance"][length] = {
            "individual_times": [],
            "batch_times": [],
            "speedup_ratios": [],
            "chunks_per_second_individual": [],
            "chunks_per_second_batch": []
        }
        
        results["consistency"][length] = {
            "avg_distances": [],
            "max_distances": [],
            "avg_similarities": [],
            "min_similarities": []
        }
        
        for batch_size in batch_sizes:
            if batch_size > len(chunks):
                print(f"Skipping batch size {batch_size} as it exceeds available chunks ({len(chunks)})")
                continue
                
            print(f"\nTesting batch size: {batch_size}")
            
            # Limit to the current batch size for testing
            test_chunks = chunks[:batch_size]
            num_chunks = len(test_chunks)
            
            # PERFORMANCE TESTING
            print("  Measuring performance...")
            
            # Individual embedding timing (multiple trials)
            individual_times = []
            for trial in range(num_trials):
                start_time = time.time()
                # Process each chunk individually
                for chunk in test_chunks:
                    _ = embed_string(chunk)
                end_time = time.time()
                trial_time = end_time - start_time
                individual_times.append(trial_time)
            
            avg_individual_time = np.mean(individual_times)
            
            # Batch embedding timing (multiple trials)
            batch_times = []
            for trial in range(num_trials):
                start_time = time.time()
                # Process all chunks in one batch
                _ = embed_list(test_chunks)
                end_time = time.time()
                trial_time = end_time - start_time
                batch_times.append(trial_time)
            
            avg_batch_time = np.mean(batch_times)
            
            # Calculate speedup and throughput
            speedup = avg_individual_time / avg_batch_time
            individual_throughput = num_chunks / avg_individual_time
            batch_throughput = num_chunks / avg_batch_time
            
            print(f"    Individual: {avg_individual_time:.2f}s, Batch: {avg_batch_time:.2f}s")
            print(f"    Speedup: {speedup:.2f}x, Batch throughput: {batch_throughput:.2f} chunks/sec")
            
            # Store performance results
            results["performance"][length]["individual_times"].append(avg_individual_time)
            results["performance"][length]["batch_times"].append(avg_batch_time)
            results["performance"][length]["speedup_ratios"].append(speedup)
            results["performance"][length]["chunks_per_second_individual"].append(individual_throughput)
            results["performance"][length]["chunks_per_second_batch"].append(batch_throughput)
            
            # CONSISTENCY TESTING
            print("  Measuring consistency...")
            
            # Embed each chunk individually
            singles = np.array([embed_string(s) for s in test_chunks])
            # Embed all chunks in a batch
            as_list = embed_list(test_chunks)
            
            # Calculate Euclidean distances
            distances = []
            for single_embedding, as_list_embedding in zip(singles, as_list):
                distance = np.sqrt(((single_embedding - as_list_embedding) ** 2).sum())
                distances.append(distance)
            
            distances = np.array(distances)
            avg_distance = np.mean(distances)
            max_distance = np.max(distances)
            
            # Calculate cosine similarities
            similarities = []
            for single_embedding, as_list_embedding in zip(singles, as_list):
                vector1 = single_embedding.reshape(1, -1)
                vector2 = as_list_embedding.reshape(1, -1)
                similarity = cosine_similarity(vector1, vector2)[0][0]
                similarities.append(similarity)
            
            similarities = np.array(similarities)
            avg_similarity = np.mean(similarities)
            min_similarity = np.min(similarities)
            
            print(f"    Avg Euclidean distance: {avg_distance:.4f}, Max: {max_distance:.4f}")
            print(f"    Avg Cosine similarity: {avg_similarity:.4f}, Min: {min_similarity:.4f}")
            
            # Store consistency results
            results["consistency"][length]["avg_distances"].append(avg_distance)
            results["consistency"][length]["max_distances"].append(max_distance)
            results["consistency"][length]["avg_similarities"].append(avg_similarity)
            results["consistency"][length]["min_similarities"].append(min_similarity)
    
    return results

def plot_comparative_results(all_results, text_lengths, batch_sizes):
    """Create visualizations comparing different embedding models"""
    models = list(all_results.keys())
    
    # Create a figure for model comparison
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(f'Ollama Embedding Model Comparison\n'
                f'OLLAMA_NUM_PARALLEL: {os.environ.get("OLLAMA_NUM_PARALLEL", "1")}', 
                fontsize=16)
    
    # Plot layout - 4 rows, 2 columns
    gs = fig.add_gridspec(4, 2)
    
    # 1. Speedup Comparison - Short Text
    ax1 = fig.add_subplot(gs[0, 0])
    _plot_model_comparison(ax1, all_results, models, "speedup", 
                         text_lengths[0], "Speedup Ratio - Short Text")
    
    # 2. Speedup Comparison - Medium Text
    ax2 = fig.add_subplot(gs[0, 1])
    _plot_model_comparison(ax2, all_results, models, "speedup", 
                         text_lengths[2], "Speedup Ratio - Medium Text")
    
    # 3. Throughput Comparison - Short Text
    ax3 = fig.add_subplot(gs[1, 0])
    _plot_model_comparison(ax3, all_results, models, "throughput", 
                         text_lengths[0], "Throughput - Short Text")
    
    # 4. Throughput Comparison - Medium Text
    ax4 = fig.add_subplot(gs[1, 1])
    _plot_model_comparison(ax4, all_results, models, "throughput", 
                         text_lengths[2], "Throughput - Medium Text")
    
    # 5. Best Batch Size by Model (Speedup)
    ax7 = fig.add_subplot(gs[2, 0])
    _plot_best_batch_size(ax7, all_results, models, text_lengths, "speedup")
    
    # 6. Best Batch Size by Model (Throughput)
    ax8 = fig.add_subplot(gs[2, 1])
    _plot_best_batch_size(ax8, all_results, models, text_lengths, "throughput")
    
    # 7. Latency Comparison by Model - Short Text
    ax9 = fig.add_subplot(gs[3, 0])
    _plot_model_comparison(ax9, all_results, models, "latency", 
                         text_lengths[0], "Latency - Short Text")
    
    # 8. Latency Comparison by Model - Medium Text
    ax10 = fig.add_subplot(gs[3, 1])
    _plot_model_comparison(ax10, all_results, models, "latency", 
                          text_lengths[2], "Latency - Medium Text")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"ollama_model_comparison_{timestamp}.png"
    plt.savefig(plot_filename, dpi=300)
    print(f"\nModel comparison visualization saved as {plot_filename}")
    
    return plot_filename

def _plot_model_comparison(ax, all_results, models, metric_type, text_length, title):
    """Helper function to plot model comparison for a specific metric and text length"""
    for model in models:
        results = all_results[model]
        
        # Find index of the text length if it exists
        if text_length not in results["text_lengths"]:
            continue
            
        # Get the index for this text length
        text_length_idx = results["text_lengths"].index(text_length)
        
        # Get the performance metrics for this length
        if metric_type == "speedup":
            metric_values = results["performance"][text_length]["speedup_ratios"]
        elif metric_type == "throughput":
            metric_values = results["performance"][text_length]["chunks_per_second_batch"]
        elif metric_type == "similarity":
            metric_values = results["consistency"][text_length]["avg_similarities"]
        elif metric_type == "latency":
            metric_values = results["performance"][text_length]["batch_times"]
        else:
            continue
        
        if len(metric_values) == 0:
            continue
        
        batch_sizes = results["batch_sizes"][:len(metric_values)]
        batch_sizes_str = [str(size) for size in batch_sizes]
        
        ax.plot(batch_sizes_str, metric_values, 'o-', label=model)
    
    ax.set_xlabel('Batch Size')
    
    if metric_type == "speedup":
        ax.set_ylabel('Speedup Ratio')
        ax.axhline(y=1, color='r', linestyle='--', alpha=0.7)
    elif metric_type == "throughput":
        ax.set_ylabel('Chunks Per Second')
    elif metric_type == "similarity":
        ax.set_ylabel('Cosine Similarity')
    elif metric_type == "latency":
        ax.set_ylabel('Processing Time (seconds)')
    
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)

def _plot_best_batch_size(ax, all_results, models, text_lengths, metric_type):
    """Plot the best batch size for each model and text length"""
    # Create data structure
    data = {}
    for model in models:
        data[model] = []
        
        for length in text_lengths:
            results = all_results[model]
            
            if length not in results["performance"]:
                data[model].append(0)
                continue
                
            # Find the best batch size
            if metric_type == "speedup":
                metric_values = results["performance"][length]["speedup_ratios"]
            elif metric_type == "throughput":
                metric_values = results["performance"][length]["chunks_per_second_batch"]
            else:
                data[model].append(0)
                continue
            
            if len(metric_values) == 0:
                data[model].append(0)
                continue
                
            best_idx = np.argmax(metric_values)
            best_batch_size = results["batch_sizes"][:len(metric_values)][best_idx]
            data[model].append(best_batch_size)
    
    # Plot as heatmap
    if not any(len(data[model]) > 0 for model in models):
        ax.text(0.5, 0.5, "Insufficient data for heatmap", 
                ha='center', va='center', transform=ax.transAxes)
        return
    
    model_data = np.array([data[model] for model in models])
    
    # Create a mask for zero values
    mask = (model_data == 0)
    model_data_masked = np.ma.array(model_data, mask=mask)
    
    im = ax.imshow(model_data_masked, cmap='viridis')
    ax.set_xticks(np.arange(len(text_lengths)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels([str(length) for length in text_lengths])
    ax.set_yticklabels(models)
    ax.set_xlabel('Text Length (chars)')
    ax.set_ylabel('Model')
    
    if metric_type == "speedup":
        ax.set_title('Best Batch Size for Speedup')
    elif metric_type == "throughput":
        ax.set_title('Best Batch Size for Throughput')
    
    # Add text annotations
    for i in range(len(models)):
        for j in range(len(text_lengths)):
            if model_data[i, j] > 0:
                text = ax.text(j, i, str(int(model_data[i, j])),
                          ha="center", va="center", color="w" if model_data[i, j] > 64 else "black")
    
    plt.colorbar(im, ax=ax)

def _plot_model_ranking(ax, all_results, models, text_lengths, metric_type):
    """Plot overall model ranking based on average performance across text lengths"""
    # Calculate average metric value for each model
    model_scores = {}
    
    for model in models:
        scores = []
        
        for length in text_lengths:
            results = all_results[model]
            
            if length not in results["performance"]:
                continue
                
            # Get the best metric value for this text length
            if metric_type == "speedup":
                metric_values = results["performance"][length]["speedup_ratios"]
            elif metric_type == "throughput":
                metric_values = results["performance"][length]["chunks_per_second_batch"]
            else:
                continue
            
            if len(metric_values) > 0:
                best_value = np.max(metric_values)
                scores.append(best_value)
        
        if scores:
            model_scores[model] = np.mean(scores)
    
    # Check if we have any scores
    if not model_scores:
        ax.text(0.5, 0.5, "Insufficient data for ranking", 
                ha='center', va='center', transform=ax.transAxes)
        return
    
    # Sort models by score
    sorted_models = sorted(model_scores.keys(), key=lambda x: model_scores[x], reverse=True)
    scores = [model_scores[model] for model in sorted_models]
    
    # Create bar chart
    y_pos = np.arange(len(sorted_models))
    ax.barh(y_pos, scores, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_models)
    ax.invert_yaxis()  # Labels read top-to-bottom
    
    if metric_type == "speedup":
        ax.set_xlabel('Average Best Speedup Ratio')
        ax.set_title('Model Ranking by Speedup')
    elif metric_type == "throughput":
        ax.set_xlabel('Average Best Throughput (chunks/s)')
        ax.set_title('Model Ranking by Throughput')
    
    # 3. Heatmap for Speedup Ratio
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Find the maximum number of batch sizes actually measured for any text length
    max_measured_batch_sizes = 0
    for length in text_lengths:
        max_measured_batch_sizes = max(max_measured_batch_sizes, 
                                       len(results["performance"][length]["speedup_ratios"]))
    
    # Use only batch sizes that were measured
    measured_batch_sizes = batch_sizes[:max_measured_batch_sizes]
    
    speedup_data = np.zeros((len(text_lengths), len(measured_batch_sizes)))
    speedup_mask = np.ones(speedup_data.shape, dtype=bool)
    
    for i, length in enumerate(text_lengths):
        ratios = results["performance"][length]["speedup_ratios"]
        for j in range(len(ratios)):
            speedup_data[i, j] = ratios[j]
            speedup_mask[i, j] = False
    
    # Create masked array for proper heatmap display
    masked_speedup = np.ma.array(speedup_data, mask=speedup_mask)
    
    im = ax3.imshow(masked_speedup, cmap='viridis')
    ax3.set_xticks(np.arange(len(measured_batch_sizes)))
    ax3.set_yticks(np.arange(len(text_lengths)))
    ax3.set_xticklabels(measured_batch_sizes)
    ax3.set_yticklabels(text_lengths)
    ax3.set_xlabel('Batch Size')
    ax3.set_ylabel('Text Length (chars)')
    ax3.set_title('Speedup Ratio Heatmap')
    
    # Add text annotations for non-masked values
    for i in range(len(text_lengths)):
        for j in range(len(measured_batch_sizes)):
            if not speedup_mask[i, j]:
                text = ax3.text(j, i, f"{speedup_data[i, j]:.1f}",
                              ha="center", va="center", color="w" if speedup_data[i, j] < 2 else "black")
    
    plt.colorbar(im, ax=ax3)
    
    # 4. Heatmap for Throughput
    ax4 = fig.add_subplot(gs[1, 1])
    
    throughput_data = np.zeros((len(text_lengths), len(measured_batch_sizes)))
    throughput_mask = np.ones(throughput_data.shape, dtype=bool)
    
    for i, length in enumerate(text_lengths):
        throughputs = results["performance"][length]["chunks_per_second_batch"]
        for j in range(len(throughputs)):
            throughput_data[i, j] = throughputs[j]
            throughput_mask[i, j] = False
    
    # Create masked array for proper heatmap display
    masked_throughput = np.ma.array(throughput_data, mask=throughput_mask)
    
    im = ax4.imshow(masked_throughput, cmap='plasma')
    ax4.set_xticks(np.arange(len(measured_batch_sizes)))
    ax4.set_yticks(np.arange(len(text_lengths)))
    ax4.set_xticklabels(measured_batch_sizes)
    ax4.set_yticklabels(text_lengths)
    ax4.set_xlabel('Batch Size')
    ax4.set_ylabel('Text Length (chars)')
    ax4.set_title('Throughput Heatmap (chunks/second)')
    
    # Calculate max throughput of non-masked values for color threshold
    max_visible_throughput = np.max(throughput_data[~throughput_mask])
    
    # Add text annotations for non-masked values
    for i in range(len(text_lengths)):
        for j in range(len(measured_batch_sizes)):
            if not throughput_mask[i, j]:
                text = ax4.text(j, i, f"{throughput_data[i, j]:.1f}",
                              ha="center", va="center", 
                              color="w" if throughput_data[i, j] < max_visible_throughput/2 else "black")
    
    plt.colorbar(im, ax=ax4)
    
    # 5. Consistency - Euclidean Distance by Text Length
    ax5 = fig.add_subplot(gs[2, 0])
    for length in text_lengths:
        length_str = str(length)
        if length_str in cons_data and len(cons_data[length_str]['distance']) > 0:
            ax5.plot(cons_data[length_str]['batch_sizes_str'], 
                    cons_data[length_str]['distance'], 
                    'o-', label=f'{length} chars')
    
    ax5.set_xlabel('Batch Size')
    ax5.set_ylabel('Average Euclidean Distance')
    ax5.set_title('Embedding Consistency (Euclidean) by Text Length')
    ax5.legend(title='Text Length')
    ax5.grid(True, linestyle='--', alpha=0.7)
    
    # 6. Consistency - Cosine Similarity by Text Length
    ax6 = fig.add_subplot(gs[2, 1])
    for length in text_lengths:
        length_str = str(length)
        if length_str in cons_data and len(cons_data[length_str]['similarity']) > 0:
            ax6.plot(cons_data[length_str]['batch_sizes_str'], 
                    cons_data[length_str]['similarity'], 
                    'o-', label=f'{length} chars')
    
    ax6.set_xlabel('Batch Size')
    ax6.set_ylabel('Average Cosine Similarity')
    ax6.set_title('Embedding Consistency (Cosine) by Text Length')
    ax6.legend(title='Text Length')
    ax6.grid(True, linestyle='--', alpha=0.7)
    
    # 7. Batch Size Recommendations
    ax7 = fig.add_subplot(gs[3, 0])
    
    # Calculate optimal batch sizes
    optimal_speedup = {}
    optimal_throughput = {}
    optimal_consistency = {}
    
    for length in text_lengths:
        # Find batch size with highest speedup
        if len(results["performance"][length]["speedup_ratios"]) > 0:
            speedup_values = results["performance"][length]["speedup_ratios"]
            best_speedup_idx = np.argmax(speedup_values)
            optimal_speedup[length] = batch_sizes[:len(speedup_values)][best_speedup_idx]
            
            # Find batch size with highest throughput
            throughput_values = results["performance"][length]["chunks_per_second_batch"]
            best_throughput_idx = np.argmax(throughput_values)
            optimal_throughput[length] = batch_sizes[:len(throughput_values)][best_throughput_idx]
            
            # Find batch size with best consistency (highest cosine similarity)
            if len(results["consistency"][length]["avg_similarities"]) > 0:
                similarity_values = results["consistency"][length]["avg_similarities"]
                best_consistency_idx = np.argmax(similarity_values)
                optimal_consistency[length] = batch_sizes[:len(similarity_values)][best_consistency_idx]
    
    # Prepare data for bar chart
    bar_data = []
    bar_labels = []
    
    for length in text_lengths:
        if length in optimal_speedup and length in optimal_throughput and length in optimal_consistency:
            bar_data.append([optimal_speedup[length], optimal_throughput[length], optimal_consistency[length]])
            bar_labels.append(str(length))
    
    if bar_data:
        bar_data = np.array(bar_data).T
        width = 0.25
        x = np.arange(len(bar_labels))
        
        ax7.bar(x - width, bar_data[0], width, label='Best Speedup')
        ax7.bar(x, bar_data[1], width, label='Best Throughput')
        ax7.bar(x + width, bar_data[2], width, label='Best Consistency')
        
        ax7.set_xlabel('Text Length (chars)')
        ax7.set_ylabel('Optimal Batch Size')
        ax7.set_title('Recommended Batch Sizes by Optimization Goal')
        ax7.set_xticks(x)
        ax7.set_xticklabels(bar_labels)
        ax7.legend()
    else:
        ax7.text(0.5, 0.5, "Insufficient data for recommendations", 
                ha='center', va='center', transform=ax7.transAxes)
    
    # 8. Batch Size vs Latency
    ax8 = fig.add_subplot(gs[3, 1])
    
    for length in text_lengths:
        length_str = str(length)
        if length_str in perf_data and 'batch_sizes_str' in perf_data[length_str]:
            batch_times = results["performance"][length]["batch_times"]
            if len(batch_times) > 0:
                ax8.plot(perf_data[length_str]['batch_sizes_str'], 
                        batch_times, 
                        'o-', label=f'{length} chars')
    
    ax8.set_xlabel('Batch Size')
    ax8.set_ylabel('Processing Time (seconds)')
    ax8.set_title('Batch Processing Latency by Text Length')
    ax8.legend(title='Text Length')
    ax8.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"ollama_text_length_benchmark_{timestamp}.png"
    plt.savefig(plot_filename, dpi=300)
    print(f"\nVisualization saved as {plot_filename}")
    
    # Save raw results as JSON
    results_filename = f"ollama_text_length_benchmark_{timestamp}.json"
    with open(results_filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Raw results saved as {results_filename}")
    
    return plot_filename

def analyze_comparative_results(all_results, text_lengths, batch_sizes):
    """Generate a text summary comparing the performance of different embedding models"""
    models = list(all_results.keys())
    
    # Calculate key metrics for each model
    model_metrics = {}
    
    for model in models:
        results = all_results[model]
        
        # Initialize metric storage
        model_metrics[model] = {
            "avg_speedup": [],
            "max_speedup": [],
            "avg_throughput": [],
            "max_throughput": [],
            "avg_latency": [],
            "min_latency": [],
            "best_batch_sizes": {}
        }
        
        # Calculate metrics across text lengths
        for length in text_lengths:
            if length not in results["performance"]:
                continue
                
            # Get performance metrics
            speedup = results["performance"][length]["speedup_ratios"]
            throughput = results["performance"][length]["chunks_per_second_batch"]
            latency = results["performance"][length]["batch_times"]
            
            if len(speedup) > 0:
                model_metrics[model]["avg_speedup"].append(np.mean(speedup))
                model_metrics[model]["max_speedup"].append(np.max(speedup))
                
                # Find best batch size for speedup
                best_speedup_idx = np.argmax(speedup)
                best_speedup_size = results["batch_sizes"][:len(speedup)][best_speedup_idx]
                
                if length not in model_metrics[model]["best_batch_sizes"]:
                    model_metrics[model]["best_batch_sizes"][length] = {}
                    
                model_metrics[model]["best_batch_sizes"][length]["speedup"] = best_speedup_size
            
            if len(throughput) > 0:
                model_metrics[model]["avg_throughput"].append(np.mean(throughput))
                model_metrics[model]["max_throughput"].append(np.max(throughput))
                
                # Find best batch size for throughput
                best_throughput_idx = np.argmax(throughput)
                best_throughput_size = results["batch_sizes"][:len(throughput)][best_throughput_idx]
                
                if length not in model_metrics[model]["best_batch_sizes"]:
                    model_metrics[model]["best_batch_sizes"][length] = {}
                    
                model_metrics[model]["best_batch_sizes"][length]["throughput"] = best_throughput_size
            
            if len(latency) > 0:
                model_metrics[model]["avg_latency"].append(np.mean(latency))
                model_metrics[model]["min_latency"].append(np.min(latency))
    
    # Generate summary text
    summary = [
        "# OLLAMA EMBEDDING MODEL COMPARISON",
        f"OLLAMA_NUM_PARALLEL: {os.environ.get('OLLAMA_NUM_PARALLEL', '1')}",
        f"Models tested: {', '.join(models)}",
        "",
        "## Overall Model Performance Ranking",
        ""
    ]
    
    # Rank models by average max throughput
    throughput_ranking = []
    for model in models:
        if model_metrics[model]["max_throughput"]:
            avg_max_throughput = np.mean(model_metrics[model]["max_throughput"])
            throughput_ranking.append((model, avg_max_throughput))
    
    throughput_ranking.sort(key=lambda x: x[1], reverse=True)
    
    summary.append("### By Throughput (chunks/second)")
    summary.append("")
    for i, (model, score) in enumerate(throughput_ranking, 1):
        summary.append(f"{i}. **{model}**: {score:.2f} chunks/s average maximum throughput")
    
    summary.append("")
    
    # Rank models by average max speedup
    speedup_ranking = []
    for model in models:
        if model_metrics[model]["max_speedup"]:
            avg_max_speedup = np.mean(model_metrics[model]["max_speedup"])
            speedup_ranking.append((model, avg_max_speedup))
    
    speedup_ranking.sort(key=lambda x: x[1], reverse=True)
    
    summary.append("### By Speedup Ratio")
    summary.append("")
    for i, (model, score) in enumerate(speedup_ranking, 1):
        summary.append(f"{i}. **{model}**: {score:.2f}x average maximum speedup")
    
    summary.append("")
    summary.append("## Optimal Batch Sizes by Text Length")
    summary.append("")
    
    # Create table for short queries
    summary.append("### For Short Queries (15 chars)")
    summary.append("")
    summary.append("| Model | Best Speedup Batch Size | Best Throughput Batch Size | Latency (ms) |")
    summary.append("|-------|-------------------------|----------------------------|--------------|")
    
    for model in models:
        metrics = model_metrics[model]
        
        speedup_size = "N/A"
        throughput_size = "N/A"
        latency_ms = "N/A"
        
        if 15 in metrics["best_batch_sizes"]:
            if "speedup" in metrics["best_batch_sizes"][15]:
                speedup_size = str(metrics["best_batch_sizes"][15]["speedup"])
            
            if "throughput" in metrics["best_batch_sizes"][15]:
                throughput_size = str(metrics["best_batch_sizes"][15]["throughput"])
                
                # Find corresponding latency if available
                if 15 in all_results[model]["performance"]:
                    throughput_idx = all_results[model]["batch_sizes"].index(metrics["best_batch_sizes"][15]["throughput"])
                    if throughput_idx < len(all_results[model]["performance"][15]["batch_times"]):
                        latency_s = all_results[model]["performance"][15]["batch_times"][throughput_idx]
                        latency_ms = f"{latency_s * 1000:.1f}"
        
        summary.append(f"| {model} | {speedup_size} | {throughput_size} | {latency_ms} |")
    
    summary.append("")
    summary.append("### For Discussion Content (256 chars)")
    summary.append("")
    summary.append("| Model | Best Speedup Batch Size | Best Throughput Batch Size | Latency (ms) |")
    summary.append("|-------|-------------------------|----------------------------|--------------|")
    
    for model in models:
        metrics = model_metrics[model]
        
        speedup_size = "N/A"
        throughput_size = "N/A"
        latency_ms = "N/A"
        
        if 256 in metrics["best_batch_sizes"]:
            if "speedup" in metrics["best_batch_sizes"][256]:
                speedup_size = str(metrics["best_batch_sizes"][256]["speedup"])
            
            if "throughput" in metrics["best_batch_sizes"][256]:
                throughput_size = str(metrics["best_batch_sizes"][256]["throughput"])
                
                # Find corresponding latency if available
                if 256 in all_results[model]["performance"]:
                    throughput_idx = all_results[model]["batch_sizes"].index(metrics["best_batch_sizes"][256]["throughput"])
                    if throughput_idx < len(all_results[model]["performance"][256]["batch_times"]):
                        latency_s = all_results[model]["performance"][256]["batch_times"][throughput_idx]
                        latency_ms = f"{latency_s * 1000:.1f}"
        
        summary.append(f"| {model} | {speedup_size} | {throughput_size} | {latency_ms} |")
    
    summary.append("")
    summary.append("## Key Findings")
    summary.append("")
    
    # Extract key findings from the data
    best_model_throughput = throughput_ranking[0][0] if throughput_ranking else "N/A"
    best_model_speedup = speedup_ranking[0][0] if speedup_ranking else "N/A"
    
    summary.append(f"1. **Best Overall Model for Throughput**: {best_model_throughput}")
    summary.append(f"2. **Best Overall Model for Batch Efficiency**: {best_model_speedup}")
    
    # Find patterns in batch size recommendations
    batch_patterns = {}
    for model in models:
        batch_patterns[model] = {
            "speedup": [],
            "throughput": []
        }
        
        for length in text_lengths:
            if length in model_metrics[model]["best_batch_sizes"]:
                if "speedup" in model_metrics[model]["best_batch_sizes"][length]:
                    batch_patterns[model]["speedup"].append(model_metrics[model]["best_batch_sizes"][length]["speedup"])
                
                if "throughput" in model_metrics[model]["best_batch_sizes"][length]:
                    batch_patterns[model]["throughput"].append(model_metrics[model]["best_batch_sizes"][length]["throughput"])
    
    # Analyze patterns
    summary.append("")
    summary.append("3. **Batch Size Patterns:**")
    
    for model in models:
        speedup_sizes = batch_patterns[model]["speedup"]
        throughput_sizes = batch_patterns[model]["throughput"]
        
        if speedup_sizes:
            most_common_speedup = max(set(speedup_sizes), key=speedup_sizes.count)
            summary.append(f"   - {model}: Most common optimal batch size for speedup is {most_common_speedup}")
        
        if throughput_sizes:
            most_common_throughput = max(set(throughput_sizes), key=throughput_sizes.count)
            summary.append(f"   - {model}: Most common optimal batch size for throughput is {most_common_throughput}")
    
    summary.append("")
    summary.append("## Recommendations for BitIQ Nostr_AI Service")
    summary.append("")
    
    # Generate specific recommendations for the different use cases
    summary.append("### For Search Queries:")
    summary.append("")
    for model in models:
        if 15 in model_metrics[model]["best_batch_sizes"] and "throughput" in model_metrics[model]["best_batch_sizes"][15]:
            best_size = model_metrics[model]["best_batch_sizes"][15]["throughput"]
            summary.append(f"- If using **{model}**, use batch size {best_size} for search queries")
    
    summary.append("")
    summary.append("### For Discussion Content:")
    summary.append("")
    for model in models:
        if 256 in model_metrics[model]["best_batch_sizes"] and "throughput" in model_metrics[model]["best_batch_sizes"][256]:
            best_size = model_metrics[model]["best_batch_sizes"][256]["throughput"]
            summary.append(f"- If using **{model}**, use batch size {best_size} for discussion content")
    
    summary.append("")
    summary.append("### System Configuration:")
    summary.append("")
    summary.append(f"1. Set `OLLAMA_NUM_PARALLEL=1` for all models")
    summary.append(f"2. For priority queue implementation:")
    summary.append(f"   - Use separate processing pools for search queries and discussion content")
    
    best_query_model = best_model_throughput
    best_query_size = "N/A"
    if 15 in model_metrics.get(best_query_model, {}).get("best_batch_sizes", {}):
        if "throughput" in model_metrics[best_query_model]["best_batch_sizes"][15]:
            best_query_size = model_metrics[best_query_model]["best_batch_sizes"][15]["throughput"]
    
    best_content_model = best_model_throughput
    best_content_size = "N/A"
    if 256 in model_metrics.get(best_content_model, {}).get("best_batch_sizes", {}):
        if "throughput" in model_metrics[best_content_model]["best_batch_sizes"][256]:
            best_content_size = model_metrics[best_content_model]["best_batch_sizes"][256]["throughput"]
    
    summary.append(f"   - For search queries: use smaller batch sizes ({best_query_size}) with minimal wait time (20-50ms)")
    summary.append(f"   - For discussion content: use larger batch sizes ({best_content_size}) with longer wait times (100-300ms)")
    
    return "\n".join(summary)

def main():
    # Define parameters for this benchmark
    text_lengths = [15, 100, 256, 480]  # Specific text lengths as requested
    batch_sizes = [1, 2, 8, 16, 32, 48, 64, 80]  # Specific batch sizes as requested
    
    # Define models to test
    # Can be passed via command line arguments
    import sys
    if len(sys.argv) > 1:
        models = sys.argv[1:]
        print(f"Testing models from command line: {models}")
    else:
        models = ["nomic-embed-text"]
        print(f"No models specified, using default: {models}")
    
    # Store all results
    all_results = {}
    
    for model in models:
        print(f"\n{'='*80}")
        print(f"TESTING MODEL: {model}")
        print(f"{'='*80}\n")
        
        # Set the current model as environment variable
        os.environ["EMBEDDING_MODEL"] = model
        global EMBEDDING_MODEL
        EMBEDDING_MODEL = model
        
        print(f"Starting Ollama text length benchmark with model: {EMBEDDING_MODEL}")
        print(f"Testing text lengths: {text_lengths}")
        print(f"Testing batch sizes: {batch_sizes}")
        
        # Run the benchmark for this model
        results = test_length_impact(text_lengths, batch_sizes, num_trials=2, model_name=model)
        all_results[model] = results
        
        # Save individual model results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_results_file = f"ollama_{model}_benchmark_{timestamp}.json"
        with open(model_results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults for {model} saved as {model_results_file}")
        
        # For a single model, also generate the regular visualizations
        if len(models) == 1:
            visualize_results(results)
    
    # If multiple models were tested, create comparative visualizations
    if len(models) > 1:
        # Visualize comparative results
        plot_file = plot_comparative_results(all_results, text_lengths, batch_sizes)
        
        # Analyze and print summary
        summary = analyze_comparative_results(all_results, text_lengths, batch_sizes)
        print("\n" + "="*80 + "\n")
        print(summary)
        
        # Save analysis to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = f"ollama_model_comparison_{timestamp}.md"
        with open(summary_file, 'w') as f:
            f.write(summary)
        print(f"\nComparative analysis saved as {summary_file}")
    else:
        # For a single model, use the regular analysis
        model = models[0]
        summary = analyze_results(all_results[model])
        print("\n" + "="*80 + "\n")
        print(summary)
        
        # Save analysis to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = f"ollama_{model}_analysis_{timestamp}.md"
        with open(summary_file, 'w') as f:
            f.write(summary)
        print(f"\nAnalysis saved as {summary_file}")

if __name__ == "__main__":
    main()