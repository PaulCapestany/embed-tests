# Ollama Embedding Benchmark Tools

This repository contains tools for benchmarking and analyzing the performance and consistency of Ollama's embedding functionality. These tools help optimize batch sizes and parallel processing settings for production deployments.

## Overview

Two main scripts are provided:

1. **consistency.py**: Tests embedding consistency between individual and batch processing methods
2. **performance.py**: Measures performance (speed) differences between individual and batch processing

These tools are valuable for Site Reliability Engineers, DevOps engineers, and developers working with embedding models in production environments.

## Requirements

- Python 3.7+
- Ollama (installed and running)
- The embedding model you wish to test (e.g., `nomic-embed-text`)

### Dependencies

```
pip install ollama numpy python-dotenv scikit-learn matplotlib pandas
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/ollama-benchmark.git
cd ollama-benchmark
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure Ollama is installed and running:
```bash
# Start Ollama service if not already running
ollama serve
```

4. Pull the embedding model you want to test:
```bash
ollama pull nomic-embed-text  # or any other model
```

## Usage

### 1. Consistency Testing (consistency.py)

This script measures how consistent embeddings are when processed individually versus in batches.

```bash
# Set the embedding model to test
export EMBEDDING_MODEL=nomic-embed-text

# Set the number of parallel requests Ollama can process
export OLLAMA_NUM_PARALLEL=4

# Run the consistency test
python consistency.py
```

#### What it does:

- Loads sample text from a file (`16-h.htm` by default)
- Chunks the text into segments of fixed size
- For each batch size (2, 4, 8, 16, 32, 64, 128, 256):
  - Embeds each chunk individually
  - Embeds all chunks in a single batch
  - Compares the results using Euclidean distance and cosine similarity
- Visualizes the results in four plots showing average/max distances and average/min similarities
- Saves the visualization as `embedding_batch_performance.png`

#### Interpreting results:

- Lower Euclidean distances and higher cosine similarities indicate better consistency
- Batch size 2 typically provides the best consistency
- Larger batch sizes may show degraded consistency depending on OLLAMA_NUM_PARALLEL setting

### 2. Performance Testing (performance.py)

This script measures the speed differences between individual and batch processing.

```bash
# Set the embedding model to test
export EMBEDDING_MODEL=nomic-embed-text

# Set the number of parallel requests Ollama can process
export OLLAMA_NUM_PARALLEL=8

# Run the performance test
python performance.py
```

#### What it does:

- Loads and chunks sample text
- For each batch size (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024):
  - Measures the time to process chunks individually
  - Measures the time to process chunks as a batch
  - Calculates speedup ratio and throughput metrics
- Runs multiple trials for statistical reliability
- Visualizes the results in four plots:
  - Execution time comparison
  - Batch processing speedup ratio
  - Throughput comparison (chunks per second)
  - Scaling efficiency (normalized throughput)
- Saves both the visualization and raw results as JSON

#### Interpreting results:

- Speedup ratio > 1.0 indicates batching is faster
- Throughput (chunks per second) shows practical processing capacity
- Look for the batch size where throughput plateaus (typically around 64)
- Batch sizes that provide speedup ratios of ~2x with good consistency are optimal

## Parameters

### Environment Variables

- `EMBEDDING_MODEL`: The Ollama model to use (e.g., `nomic-embed-text`, `nomic-embed-text`)
- `OLLAMA_NUM_PARALLEL`: Number of parallel requests Ollama will process

### Script-Specific Parameters

#### consistency.py:
- `chunk_size`: Size of each text chunk (default: 256)
- `batch_sizes_list`: List of batch sizes to test (default: [2, 4, 8, 16, 32, 64, 128, 256])

#### performance.py:
- `batch_sizes`: List of batch sizes to test (default: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
- `num_trials`: Number of test iterations for statistical reliability (default: 3)
- `max_chars`: Maximum text to process to limit test duration (default: 100000)

## Troubleshooting

### Common Issues:

1. **Missing module error**:
   ```
   ModuleNotFoundError: No module named 'ollama'
   ```
   Solution: Install required dependencies with `pip install ollama`

2. **Ollama service not running**:
   ```
   ConnectionRefusedError: [Errno 111] Connection refused
   ```
   Solution: Start the Ollama service with `ollama serve`

3. **Model not found**:
   ```
   Error: model '<model_name>' not found
   ```
   Solution: Pull the model with `ollama pull <model_name>`

4. **Missing input file**:
   ```
   FileNotFoundError: [Errno 2] No such file or directory: '16-h.htm'
   ```
   Solution: Provide a text file for testing or modify the script to use a different file

## Best Practices

Based on benchmark results, the following configurations are recommended:

1. **For maximum embedding consistency**:
   - Use batch size 2
   - Set OLLAMA_NUM_PARALLEL to 8 or higher

2. **For optimal performance**:
   - Use batch size 64 for maximum throughput
   - Use batch size 16 for best efficiency (speedup ratio)
   - Horizontal scaling is recommended over increasing batch size beyond 64

## License

[MIT License](LICENSE)