# Ollama Embedding Optimization for nostr_ai: Multi-Model Analysis and Implementation Strategy

## Executive Summary

This document analyzes multi-model benchmark results for the BitIQ `nostr_ai` service's embedding workflow and provides a concrete implementation strategy based on the findings. Our analysis addresses two critical constraints:

1. **Multi-Model Management**: Using multiple embedding models requires separate queue systems for each model, significantly impacting latency
2. **Sequential Processing Limitations**: Due to Ollama enforcing no parallelism for embeddings, high-priority search queries can be blocked by in-progress discussion content processing

Key findings:
- **Multi-Model**: If dual models are required, much smaller batch sizes and parallel processing are essential
- **Latency Impact**: Search queries could experience unacceptable delays (>1s) with naïve implementation in multi-model scenarios

Based on detailed latency and throughput analysis, we recommend small batch sizes, with specific queue configurations for multi-model deployments.

## Latency Analysis

We evaluated worst-case scenarios where a search query arrives immediately after a discussion batch has started processing, forcing the search query to wait for the entire batch to complete.

### Dual Model (nomic-embed-text + mxbai-embed-large) Sequential Processing

| Discussion Batch | Search Batch | Worst-Case Latency | Discussion Throughput | Within 1000ms Target |
|------------------|--------------|-------------------|----------------------|----------------------|
| 8                | 8            | 469ms             | 35 chunks/s          | ✅                   |
| 8                | 16           | 636ms             | 35 chunks/s          | ✅                   |
| 8                | 32           | 904ms             | 35 chunks/s          | ✅                   |
| 16               | 8            | 775ms             | 36 chunks/s          | ✅                   |
| 16               | 16           | 942ms             | 36 chunks/s          | ✅                   |
| 16               | 32           | 1210ms            | 36 chunks/s          | ❌                   |

**Key Insight**: With sequential dual-model processing, even the smallest discussion batch sizes (8) can cause search queries to wait 469ms before processing begins.

### Parallel Model Processing (Theoretical)

If we could process both models in parallel (separate services), we would see significantly better performance:

| Discussion Batch | Search Batch | Worst-Case Latency | Within 500ms Target |
|------------------|--------------|-------------------|---------------------|
| 8                | 8            | 315ms             | ✅                  |
| 8                | 16           | 417ms             | ✅                  |
| 16               | 8            | 533ms             | ❌                  |

**Key Insight**: Even with parallel processing, discussion batch sizes should not exceed 16 to maintain reasonable search latency in a multi-model environment.

## Recommended Implementation Strategies

### Dual Models

```yaml
ollama_num_parallel: 1
  model: "nomic-embed-text"  
     high_priority_queue:
       batch_size: 8
       min_batch_size: 1
       max_wait_time: 40ms       
     standard_queue:
       batch_size: 16
       min_batch_size: 2
       max_wait_time: 250ms
  model: "mxbai-embed-large"     
     high_priority_queue:
       batch_size: 8
       min_batch_size: 1
       max_wait_time: 40ms
     standard_queue:
       batch_size: 16
       min_batch_size: 2
       max_wait_time: 250ms
```

## Queue Architecture Design

We need to rethink our queue architecture to address the multi-model and sequential processing challenges. Here are the key components:

### Batch Processing Implementation

The batch processing logic should be structured to maintain efficiency while allowing for preemption:

```go
func (m *EmbeddingQueueManager) processHighPriorityBatch() {
    m.processingMutex.Lock()
    
    // Update processing state
    m.processingState = ProcessingState{
        isProcessing: true,
        currentPriority: HighPriority,
        startTime: time.Now(),
        canInterrupt: false, // High priority can't be interrupted
    }
    
    // Take batch from queue
    batch := m.highPriorityQueue.takeBatch(m.HighPriorityConfig.BatchSize)
    batchSize := len(batch)
    m.processingState.currentBatchSize = batchSize
    
    m.processingMutex.Unlock()
    
    // Process batch
    texts := make([]string, batchSize)
    for i, req := range batch {
        texts[i] = req.Text
    }
    
    // Measure processing time
    startTime := time.Now()
    
    // Call Ollama embedding API
    embeddings, err := m.ollamaClient.EmbedBatch(texts)
    
    // Calculate metrics
    processingTime := time.Since(startTime)
    batchLatency := processingTime
    
    // Update metrics
    m.metrics.recordBatchProcessing(HighPriority, batchSize, processingTime)
    
    // Distribute results
    for i, req := range batch {
        latency := time.Since(req.Timestamp)
        m.metrics.recordRequestLatency(HighPriority, latency)
        
        result := &EmbeddingResult{
            Embedding: nil,
            Error: err,
        }
        
        if err == nil && i < len(embeddings) {
            result.Embedding = embeddings[i]
        }
        
        // Send result back to requester
        select {
        case req.ResultCh <- result:
        default:
            // Requester may have timed out
            logger.Warn("Failed to send result, channel buffer full or closed")
        }
        close(req.ResultCh)
    }
    
    // Mark as done processing
    m.processingMutex.Lock()
    m.processingState.isProcessing = false
    m.processingMutex.Unlock()
}
```

## Monitoring & Observability

Add comprehensive monitoring to track embedding performance and queue behavior:

### Key Metrics to Track

1. **Queue Health**:
   - Queue depths for both priority levels
   - Time spent waiting in queue
   - Batch fill rates (actual/target)
   - Number of interruptions per minute

2. **Processing Metrics**:
   - Batch processing time by text length and priority
   - End-to-end latency (from request to response)
   - Throughput (requests/second) for each priority level

3. **Ollama Instance Metrics**:
   - CPU and memory utilization
   - Model load time
   - Error rates
   - Response time distribution

### Suggested Prometheus Gauges and Counters

```go
// Metric initialization
func initMetrics() *EmbeddingMetrics {
    return &EmbeddingMetrics{
        QueueDepth: promauto.NewGaugeVec(
            prometheus.GaugeOpts{
                Name: "nostr_ai_embedding_queue_depth",
                Help: "Current number of items in embedding queues",
            },
            []string{"priority"},
        ),
        BatchSize: promauto.NewHistogramVec(
            prometheus.HistogramOpts{
                Name:    "nostr_ai_embedding_batch_size",
                Help:    "Distribution of batch sizes processed",
                Buckets: []float64{1, 2, 4, 8, 16, 32, 48, 64, 80, 96, 128},
            },
            []string{"priority"},
        ),
        ProcessingTime: promauto.NewHistogramVec(
            prometheus.HistogramOpts{
                Name:    "nostr_ai_embedding_processing_time_seconds",
                Help:    "Time taken to process embedding batches",
                Buckets: prometheus.ExponentialBuckets(0.01, 2, 10),
            },
            []string{"priority"},
        ),
        QueueLatency: promauto.NewHistogramVec(
            prometheus.HistogramOpts{
                Name:    "nostr_ai_embedding_queue_latency_seconds",
                Help:    "Time items spend waiting in queue",
                Buckets: prometheus.ExponentialBuckets(0.001, 2, 12),
            },
            []string{"priority"},
        ),
        Interruptions: promauto.NewCounter(
            prometheus.CounterOpts{
                Name: "nostr_ai_embedding_standard_interruptions_total",
                Help: "Number of times standard processing was interrupted",
            },
        ),
    }
}
```

### Suggested Alerting Rules

```yaml
groups:
- name: nostr_ai_embedding_alerts
  rules:
  - alert: EmbeddingHighLatency
    expr: histogram_quantile(0.95, sum(rate(nostr_ai_embedding_queue_latency_seconds_bucket{priority="high"}[5m])) by (le)) > 0.5
    for: 2m
    annotations:
      summary: "High latency for search query embeddings"
      description: "95th percentile latency for search queries exceeds 500ms"

  - alert: EmbeddingQueueBacklog
    expr: nostr_ai_embedding_queue_depth{priority="standard"} > 1000
    for: 5m
    annotations:
      summary: "Large backlog in standard embedding queue"
      description: "Standard queue has >1000 items waiting for processing"
      
  - alert: EmbeddingErrorRate
    expr: rate(nostr_ai_embedding_errors_total[5m]) / rate(nostr_ai_embedding_requests_total[5m]) > 0.05
    for: 2m
    annotations:
      summary: "High error rate in embedding requests"
      description: "Error rate exceeds 5% over 2 minutes"
```

## Recommended Next Steps

1. **Implement Dual-Queue System**:
   - Create separate high-priority and standard queues per model (nomic-embed-text and mxbai-embed-large) which stay in lockstep (just, sequentially, since it's pointless to issue requests concurrently)
   - Add preemption logic for search queries
   - Implement appropriate timeout controls for each queue

2. **Add Comprehensive Monitoring**:
   - Deploy the suggested Prometheus metrics
   - Set up Grafana dashboards to visualize queue performance
   - Implement alerting for queue depth and latency issues

By implementing these recommendations, the BitIQ nostr_ai service should achieve optimal performance for both search queries and discussion content processing, with search queries receiving the priority treatment required for excellent user experience.