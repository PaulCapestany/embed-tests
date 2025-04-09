# OLLAMA EMBEDDING MODEL COMPARISON
OLLAMA_NUM_PARALLEL: 1
Models tested: nomic-embed-text, mxbai-embed-large, snowflake-arctic-embed2, bge-m3

## Overall Model Performance Ranking

### By Throughput (chunks/second)

1. **nomic-embed-text**: 122.33 chunks/s average maximum throughput
2. **mxbai-embed-large**: 59.43 chunks/s average maximum throughput
3. **bge-m3**: 56.99 chunks/s average maximum throughput
4. **snowflake-arctic-embed2**: 56.11 chunks/s average maximum throughput

### By Speedup Ratio

1. **bge-m3**: 4.08x average maximum speedup
2. **snowflake-arctic-embed2**: 4.07x average maximum speedup
3. **nomic-embed-text**: 2.53x average maximum speedup
4. **mxbai-embed-large**: 1.63x average maximum speedup

## Optimal Batch Sizes by Text Length

### For Short Queries (15 chars)

| Model | Best Speedup Batch Size | Best Throughput Batch Size | Latency (ms) |
|-------|-------------------------|----------------------------|--------------|
| nomic-embed-text | 1 | 64 | 371.9 |
| mxbai-embed-large | 1 | 64 | 661.6 |
| snowflake-arctic-embed2 | 80 | 80 | 949.6 |
| bge-m3 | 80 | 80 | 914.1 |

### For Discussion Content (256 chars)

| Model | Best Speedup Batch Size | Best Throughput Batch Size | Latency (ms) |
|-------|-------------------------|----------------------------|--------------|
| nomic-embed-text | 80 | 80 | 770.2 |
| mxbai-embed-large | 64 | 64 | 1444.7 |
| snowflake-arctic-embed2 | 80 | 80 | 1761.8 |
| bge-m3 | 64 | 64 | 1378.9 |

## Key Findings

1. **Best Overall Model for Throughput**: nomic-embed-text
2. **Best Overall Model for Batch Efficiency**: bge-m3

3. **Batch Size Patterns:**
   - nomic-embed-text: Most common optimal batch size for speedup is 80
   - nomic-embed-text: Most common optimal batch size for throughput is 80
   - mxbai-embed-large: Most common optimal batch size for speedup is 80
   - mxbai-embed-large: Most common optimal batch size for throughput is 64
   - snowflake-arctic-embed2: Most common optimal batch size for speedup is 80
   - snowflake-arctic-embed2: Most common optimal batch size for throughput is 80
   - bge-m3: Most common optimal batch size for speedup is 80
   - bge-m3: Most common optimal batch size for throughput is 80

## Recommendations for BitIQ Nostr_AI Service

### For Search Queries:

- If using **nomic-embed-text**, use batch size 64 for search queries
- If using **mxbai-embed-large**, use batch size 64 for search queries
- If using **snowflake-arctic-embed2**, use batch size 80 for search queries
- If using **bge-m3**, use batch size 80 for search queries

### For Discussion Content:

- If using **nomic-embed-text**, use batch size 80 for discussion content
- If using **mxbai-embed-large**, use batch size 64 for discussion content
- If using **snowflake-arctic-embed2**, use batch size 80 for discussion content
- If using **bge-m3**, use batch size 64 for discussion content

### System Configuration:

1. Set `OLLAMA_NUM_PARALLEL=1` for all models
2. For priority queue implementation:
   - Use separate processing pools for search queries and discussion content
   - For search queries: use smaller batch sizes (64) with minimal wait time (20-50ms)
   - For discussion content: use larger batch sizes (80) with longer wait times (100-300ms)