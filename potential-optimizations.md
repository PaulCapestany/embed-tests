# Ollama Embedding Optimization for nostr_ai

## Background

This document summarizes optimization findings for Ollama embedding services based on extensive performance and consistency testing. These recommendations are specifically tailored for the BitIQ `nostr_ai` Golang service that processes vector embeddings for Nostr content.

## Testing Summary

We conducted two types of tests on Ollama's embedding functionality:

1. **Consistency Testing**: Measured how similar embeddings are when processed individually vs. in batches
2. **Performance Testing**: Measured speed differences between individual and batch processing

Testing was performed with various `OLLAMA_NUM_PARALLEL` settings and batch sizes ranging from 1 to 1024, with detailed timing data for each configuration.

## Key Findings

1. **Batch Processing Effectiveness**:
   - Batch processing provides ~2x throughput improvement over individual processing
   - Optimal performance occurs at batch sizes between 16-64
   - Performance degrades with batch sizes >64

2. **Consistency Considerations**:
   - Testing conclusively showed that `OLLAMA_NUM_PARALLEL=1` provides perfect embedding consistency
   - With `OLLAMA_NUM_PARALLEL=1`, consistency is maintained across all batch sizes
   - Any increase in `OLLAMA_NUM_PARALLEL` resulted in significant degradation of embedding consistency
   - This contradicts standard parallelism assumptions but was clearly demonstrated in our tests

3. **Resource Utilization**:
   - Peak efficiency (best speedup ratio) typically occurs at batch size 16
   - Maximum throughput typically occurs at batch size 64
   - Beyond batch size 64, diminishing or negative returns are observed

## Optimized Configuration for nostr_ai

### Core Recommendations

1. **Ollama Service Configuration**:
   - Set `OLLAMA_NUM_PARALLEL=1` for the Ollama service
   - Our testing conclusively demonstrated that this setting provides the most consistent embeddings
   - Higher values of OLLAMA_NUM_PARALLEL resulted in significant embedding inconsistency

2. **Batch Size Strategy**:
   - **High-throughput mode**: Use batch size 64
   - **Balanced mode**: Use batch size 16 (better efficiency and consistency)
   - **High-precision mode**: Use batch size 2 (highest consistency)

3. **Dynamic Batching**:
   - Implement adaptive batch sizes based on load conditions while keeping `OLLAMA_NUM_PARALLEL=1`:
     - Default to batch=16 during normal operation
     - Scale up to batch=64 during high-load periods
     - Any batch size can be used with similar consistency as long as `OLLAMA_NUM_PARALLEL=1` is maintained

## Implementation Guidelines for nostr_ai

### Go Implementation Considerations

```go
// Configuration parameters with timeout controls
type EmbeddingConfig struct {
    OllamaEndpoint      string
    ModelName           string
    MaxBatchSize        int           // Maximum size to ever batch (64 recommended)
    MaxBatchWaitTime    time.Duration // Maximum time to wait before processing (e.g., 50ms)
    RequestTimeout      time.Duration
    MaxRetries          int
    RetryBackoff        time.Duration
}

// Batch processor with timeout to prevent excessive waiting
type BatchProcessor struct {
    config           EmbeddingConfig
    queue            []EmbeddingRequest
    mu               sync.Mutex
    timer            *time.Timer
    processingBatch  bool
}

// Add item to batch queue with smart processing logic
func (p *BatchProcessor) QueueForProcessing(ctx context.Context, text string) ([]float32, error) {
    responseCh := make(chan EmbeddingResponse, 1)
    
    request := EmbeddingRequest{
        Text:      text,
        Response:  responseCh,
        Timestamp: time.Now(),
    }
    
    p.mu.Lock()
    
    // Add to queue
    p.queue = append(p.queue, request)
    queueSize := len(p.queue)
    
    // Process immediately if:
    // 1. We've reached optimal batch size
    // 2. We're not already processing a batch
    if queueSize >= p.config.MaxBatchSize && !p.processingBatch {
        p.processBatch()
    } else if queueSize == 1 && !p.processingBatch {
        // Start timer for first item added to empty queue
        p.timer = time.AfterFunc(p.config.MaxBatchWaitTime, func() {
            p.mu.Lock()
            defer p.mu.Unlock()
            
            if len(p.queue) > 0 && !p.processingBatch {
                p.processBatch()
            }
        })
    }
    
    p.mu.Unlock()
    
    // Wait for response with timeout
    select {
    case resp := <-responseCh:
        if resp.Error != nil {
            return nil, resp.Error
        }
        return resp.Embedding, nil
    case <-ctx.Done():
        return nil, ctx.Err()
    }
}

// Process current batch
func (p *BatchProcessor) processBatch() {
    if len(p.queue) == 0 {
        return
    }
    
    // Stop the timer if it's running
    if p.timer != nil {
        p.timer.Stop()
    }
    
    p.processingBatch = true
    
    // Calculate request latency metrics
    oldestRequest := p.queue[0].Timestamp
    queueLatency := time.Since(oldestRequest)
    
    // Take the current queue for processing
    currentBatch := p.queue
    p.queue = nil
    
    // Process asynchronously
    go func(batch []EmbeddingRequest) {
        startTime := time.Now()
        texts := make([]string, len(batch))
        
        for i, req := range batch {
            texts[i] = req.Text
        }
        
        // Call Ollama for embedding
        embeddings, err := p.embedBatch(texts)
        
        // Calculate processing metrics
        processingTime := time.Since(startTime)
        batchSize := len(batch)
        
        // Log performance metrics
        log.Printf("Batch metrics: size=%d, queue_wait=%v, processing_time=%v, total_latency=%v",
            batchSize, queueLatency, processingTime, queueLatency+processingTime)
        
        // Distribute results back to waiters
        for i, req := range batch {
            resp := EmbeddingResponse{
                Error: err,
            }
            
            if err == nil && i < len(embeddings) {
                resp.Embedding = embeddings[i]
            }
            
            req.Response <- resp
            close(req.Response)
        }
        
        p.mu.Lock()
        p.processingBatch = false
        
        // Check if new items arrived while processing
        if len(p.queue) > 0 {
            p.processBatch() // Process next batch immediately
        }
        p.mu.Unlock()
    }(currentBatch)
}
```

### Latency vs Throughput Optimization

Based on our performance testing, here are key timing considerations:

1. **Batch Size Timing Tradeoffs**:
   - Batch size 2: ~0.05s processing time for 2 chunks = 40 chunks/sec
   - Batch size 16: ~0.2s processing time for 16 chunks = 80 chunks/sec
   - Batch size 64: ~0.6s processing time for 64 chunks = 105 chunks/sec

2. **Maximum Wait Time Recommendations**:
   - **Low-latency mode**: 20-50ms maximum wait time
   - **Balanced mode**: 100-200ms maximum wait time
   - **High-throughput mode**: 300-500ms maximum wait time

3. **Dynamic Timeout Adjustment**:
   - Monitor queue growth rate
   - Reduce wait time when queue is growing rapidly
   - Increase wait time (up to maximum) when queue is stable

## Deployment & Scaling Strategy

1. **Resource Allocation**:
   - Ensure adequate memory for your chosen batch size (larger batches use more memory)
   - Monitor GPU memory usage if applicable

2. **Horizontal Scaling**:
   - Since `OLLAMA_NUM_PARALLEL` must remain at 1 for consistency, horizontal scaling is essential
   - Deploy multiple Ollama instances, each with `OLLAMA_NUM_PARALLEL=1`
   - Use a load balancer to distribute embedding requests between instances

3. **Health Checks**:
   - Implement embedding-specific health checks to detect when Ollama is returning degraded results
   - Periodically compare embeddings of reference texts to ensure consistency

## Next Steps for nostr_ai Optimization

1. **Implement the timeout-based batch processor**:
   - Use the provided Go code as a starting point
   - Adapt to your existing architecture
   - Adjust timeouts based on your latency requirements

2. **Experiment with wait time settings**:
   - Test with different MaxBatchWaitTime values (50ms, 100ms, 200ms)
   - Measure both throughput and end-to-end latency 
   - Find the optimal balance for your specific workload patterns

3. **Add detailed monitoring**:
   - Track queue wait times separately from processing times
   - Monitor batch fill percentages (actual batch size vs. max batch size)
   - Alert on excessive queue buildup

4. **Optimize for different workload patterns**:
   - For steady traffic: Use longer wait times to build larger batches
   - For bursty traffic: Use shorter wait times to process quickly, then optimize during bursts

5. **Consider rate-limiting strategies**:
   - If upstream system produces bursts that overwhelm Ollama, add a rate limiter
   - Ensure back-pressure is appropriately handled with non-blocking queues

This document provides practical implementation guidance for optimizing the `nostr_ai` service based on our Ollama embedding performance analysis, with specific focus on balancing latency and throughput.