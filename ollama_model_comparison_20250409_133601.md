# OLLAMA EMBEDDING MODEL COMPARISON
OLLAMA_NUM_PARALLEL: 1
Models tested: nomic-embed-text, mxbai-embed-large

## Overall Model Performance Ranking

### By Throughput (chunks/second)

1. **nomic-embed-text**: 83.84 chunks/s average maximum throughput
2. **mxbai-embed-large**: 48.48 chunks/s average maximum throughput

### By Speedup Ratio

1. **nomic-embed-text**: 2.24x average maximum speedup
2. **mxbai-embed-large**: 1.85x average maximum speedup

## Optimal Batch Sizes by Text Length

### For Short Queries (15 chars)

| Model | Best Speedup Batch Size | Best Throughput Batch Size | Latency (ms) |
|-------|-------------------------|----------------------------|--------------|
| nomic-embed-text | 1 | 16 | 130.0 |
| mxbai-embed-large | 1 | 12 | 173.6 |

### For Discussion Content (256 chars)

| Model | Best Speedup Batch Size | Best Throughput Batch Size | Latency (ms) |
|-------|-------------------------|----------------------------|--------------|
| nomic-embed-text | 80 | 64 | 863.7 |
| mxbai-embed-large | 8 | 8 | 203.3 |

## Key Findings

1. **Best Overall Model for Throughput**: nomic-embed-text
2. **Best Overall Model for Batch Efficiency**: nomic-embed-text

3. **Batch Size Patterns:**
   - nomic-embed-text: Most common optimal batch size for speedup is 80
   - nomic-embed-text: Most common optimal batch size for throughput is 16
   - mxbai-embed-large: Most common optimal batch size for speedup is 80
   - mxbai-embed-large: Most common optimal batch size for throughput is 8

## Recommendations for BitIQ Nostr_AI Service

### For Search Queries:

- If using **nomic-embed-text**, use batch size 16 for search queries
- If using **mxbai-embed-large**, use batch size 12 for search queries

### For Discussion Content:

- If using **nomic-embed-text**, use batch size 64 for discussion content
- If using **mxbai-embed-large**, use batch size 8 for discussion content

### System Configuration:

1. Set `OLLAMA_NUM_PARALLEL=1` for all models
2. For priority queue implementation:
   - Use separate processing pools for search queries and discussion content
   - For search queries: use smaller batch sizes (16) with minimal wait time (20-50ms)
   - For discussion content: use larger batch sizes (64) with longer wait times (100-300ms)