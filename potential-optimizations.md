# Ollama Embedding Optimization for nostr_ai: Multi-Model Analysis and Implementation Strategy

## Executive Summary

This document analyzes multi-model benchmark results for the BitIQ `nostr_ai` service's embedding workflow and provides a concrete implementation strategy based on the findings. Our analysis addresses two critical constraints:

1. **Multi-Model Management**: Using multiple embedding models requires separate queue systems for each model, significantly impacting latency
2. **Sequential Processing Limitations**: High-priority search queries can be blocked by in-progress discussion content processing

Key findings:
- **Single Model**: For optimal performance, using only **nomic-embed-text** with small discussion batch sizes (8-16) provides best balance of throughput and responsiveness
- **Multi-Model**: If dual models are required, much smaller batch sizes and parallel processing are essential
- **Latency Impact**: Search queries could experience unacceptable delays (>1s) with naïve implementation in multi-model scenarios

Based on detailed latency and throughput analysis, we recommend significant batch size reductions from previous recommendations, with specific queue configurations for single vs. multi-model deployments.

## Latency Analysis

We evaluated worst-case scenarios where a search query arrives immediately after a discussion batch has started processing, forcing the search query to wait for the entire batch to complete.

### Single Model (nomic-embed-text) Latency

| Discussion Batch | Search Batch | Worst-Case Latency | Discussion Throughput | Within 500ms Target |
|------------------|--------------|-------------------|-----------------------|---------------------|
| 8                | 8            | 154ms             | 85 chunks/s           | ✅                  |
| 8                | 16           | 219ms             | 85 chunks/s           | ✅                  |
| 8                | 32           | 322ms             | 85 chunks/s           | ✅                  |
| 8                | 64           | 466ms             | 85 chunks/s           | ✅                  |
| 16               | 8            | 242ms             | 88 chunks/s           | ✅                  |
| 16               | 16           | 307ms             | 88 chunks/s           | ✅                  |
| 16               | 32           | 410ms             | 88 chunks/s           | ✅                  |
| 32               | 8            | 400ms             | 94 chunks/s           | ✅                  |
| 32               | 16           | 465ms             | 94 chunks/s           | ✅                  |
| 64               | 8            | 694ms             | 101 chunks/s          | ❌                  |

**Key Insight**: Discussion batch sizes above 32 can cause unacceptable search query latency (>500ms), even with a single model.

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

**Key Insight**: Even with parallel processing, discussion batch sizes should not exceed 8 to maintain reasonable search latency in a multi-model environment.

## Recommended Implementation Strategies

Based on our latency analysis, we recommend two distinct approaches depending on whether you need to run one or two embedding models.

### Approach 1: Single Model (nomic-embed-text)

If quality requirements can be met with nomic-embed-text alone, this is by far the best approach for performance and simplicity.

```yaml
embedding:
  model: "nomic-embed-text"  # Significantly outperforms other models in throughput
  ollama_num_parallel: 1     # Critical for embedding consistency
  
  high_priority_queue:       # For search queries
    batch_size: 32           # Optimal balance of throughput vs. latency
    min_batch_size: 4        # Process with minimal wait if at least 4 items
    max_wait_time: 30ms      # Short timeout for interactive use
    
  standard_queue:            # For discussion content
    batch_size: 16           # Critical to keep worst-case latency under 500ms
    min_batch_size: 4        # Process small batches quickly  
    max_wait_time: 100ms     # Shorter timeout than previously recommended
```

**Performance Characteristics**:
- Worst-case search query latency: ~307ms (16 + 16 batch sizes)
- Discussion content throughput: ~88 chunks/second
- All search queries completed within 500ms, even in worst-case scenarios

### Approach 2: Dual Models with Parallel Processing

If you absolutely need both models, you should:

1. Run separate Ollama instances for each model
2. Process each model in parallel through separate microservices
3. Dramatically reduce batch sizes to manage latency

```yaml
# Configuration for nomic-embed-text service
nomic_service:
  model: "nomic-embed-text"
  ollama_num_parallel: 1
  
  high_priority_queue:
    batch_size: 16
    min_batch_size: 2
    max_wait_time: 20ms
    
  standard_queue:
    batch_size: 8             # Critical to keep under 8 for multi-model scenarios
    min_batch_size: 2
    max_wait_time: 50ms

# Configuration for mxbai-embed-large service  
mxbai_service:
  model: "mxbai-embed-large"
  ollama_num_parallel: 1
  
  high_priority_queue:
    batch_size: 8
    min_batch_size: 2
    max_wait_time: 20ms
    
  standard_queue:
    batch_size: 8
    min_batch_size: 2
    max_wait_time: 50ms
```

**Performance Characteristics**:
- Parallel processing worst-case search query latency: ~417ms
- Discussion content throughput: ~35 chunks/second (limited by slower model)
- All search queries completed within 500ms in parallel processing

### Approach 3: Single Model with Micro-Batching (Alternative)

If your workload is dominated by discussion content with infrequent search queries, consider an adaptive micro-batching approach:

```yaml
embedding:
  model: "nomic-embed-text"
  ollama_num_parallel: 1
  
  high_priority_queue:
    batch_size: 32           # Optimized for search query throughput
    min_batch_size: 1        # Process immediately
    max_wait_time: 10ms      # Minimal waiting
    
  standard_queue:
    batch_size: 32           # Higher throughput during quiet periods
    min_batch_size: 16       # Begin processing with reasonable batch
    max_wait_time: 150ms     # Allow time to form batches
    
    # Micro-batching parameters
    micro_batch_size: 8      # Switch to smaller batches when search traffic appears
    search_traffic_threshold: 1 # Consider any waiting search query as high traffic
```

This approach dynamically switches to smaller batches when search queries are detected, minimizing the worst-case wait time.

## Queue Architecture Design

We need to rethink our queue architecture to address the multi-model and sequential processing challenges. Here are the key components:

### Single Model Per Instance Approach

Each model requires its own dedicated processing pipeline, with separate high-priority and standard queues:

```go
// ModelProcessingPipeline manages embedding requests for a single model
type ModelProcessingPipeline struct {
    Model              string
    HighPriorityQueue  *Queue
    StandardQueue      *Queue
    HighPriorityConfig QueueConfig
    StandardConfig     QueueConfig
    OllamaClient       OllamaClient
    Metrics            *EmbeddingMetrics
    
    processingMutex    sync.Mutex
    processingState    ProcessingState
    shutdownCh         chan struct{}
}

// ProcessingService coordinates multiple model pipelines
type ProcessingService struct {
    Pipelines          map[string]*ModelProcessingPipeline
    SearchResultsCh    chan *SearchResult
    Metrics            *GlobalMetrics
}
```

### Dynamic Batch Size Adjustment

To adapt to changing traffic patterns and ensure search queries don't wait too long:

```go
// QueueConfig defines settings for a priority queue
type QueueConfig struct {
    BaseBatchSize      int            // Default batch size during normal operation
    MinBatchSize       int            // Minimum batch size to process immediately
    MaxBatchSize       int            // Maximum batch size under any circumstances
    MaxWaitTime        time.Duration  // Maximum time to wait for batch formation
    
    // Dynamic adjustment
    MicroBatchSize     int            // Smaller batch size during high search traffic
    SearchThreshold    int            // Number of waiting search queries to trigger micro-batches
}

// Check if we should use micro-batching
func (p *ModelProcessingPipeline) getCurrentBatchSize() int {
    // If search traffic is present, use smaller batches for discussion content
    if p.HighPriorityQueue.Size() >= p.StandardConfig.SearchThreshold {
        return p.StandardConfig.MicroBatchSize
    }
    return p.StandardConfig.BaseBatchSize
}
```

### Multi-Model Coordination

When using multiple models, we need to coordinate between them for the final results:

```go
// Submit a search query to all model pipelines
func (s *ProcessingService) SubmitSearchQuery(ctx context.Context, query string) (*CombinedResult, error) {
    var wg sync.WaitGroup
    resultCh := make(chan *ModelResult, len(s.Pipelines))
    
    // Submit to each pipeline in parallel
    for modelName, pipeline := range s.Pipelines {
        wg.Add(1)
        go func(name string, p *ModelProcessingPipeline) {
            defer wg.Done()
            
            // Submit with timeout
            ctx, cancel := context.WithTimeout(ctx, 1*time.Second)
            defer cancel()
            
            result, err := p.Submit(ctx, query, HighPriority)
            modelResult := &ModelResult{
                ModelName: name,
                Embedding: result,
                Error:     err,
            }
            
            select {
            case resultCh <- modelResult:
            default:
                // Channel buffer full (shouldn't happen with properly sized channel)
                s.Metrics.DroppedResults.Inc()
            }
        }(modelName, pipeline)
    }
    
    // Create channel closer
    go func() {
        wg.Wait()
        close(resultCh)
    }()
    
    // Collect results with timeout
    timeout := time.After(1100 * time.Millisecond)
    results := make([]*ModelResult, 0, len(s.Pipelines))
    
    for {
        select {
        case result, ok := <-resultCh:
            if !ok {
                // Channel closed, all results collected
                return &CombinedResult{Results: results}, nil
            }
            results = append(results, result)
            
        case <-timeout:
            // Return partial results after timeout
            s.Metrics.TimeoutSearches.Inc()
            return &CombinedResult{
                Results: results,
                Partial: true,
                Error:   fmt.Errorf("search timeout: collected %d/%d results", 
                         len(results), len(s.Pipelines)),
            }, nil
        }
    }
}
```

### Pipeline Processing Implementation

Each pipeline manages its own queues independently:

```go
// processNextBatch decides which queue to process next for this model
func (p *ModelProcessingPipeline) processNextBatch() {
    p.processingMutex.Lock()
    defer p.processingMutex.Unlock()
    
    // If already processing, don't start another batch
    if p.processingState.isProcessing {
        return
    }
    
    // Always check high-priority queue first
    highPrioritySize := p.HighPriorityQueue.Size()
    highPriorityAge := p.HighPriorityQueue.OldestItemAge()
    
    // Process high priority queue if it has items and meets criteria
    if highPrioritySize > 0 && 
       (highPrioritySize >= p.HighPriorityConfig.MinBatchSize || 
        highPriorityAge >= p.HighPriorityConfig.MaxWaitTime) {
        
        batchSize := p.HighPriorityConfig.BaseBatchSize
        if highPrioritySize < batchSize {
            batchSize = highPrioritySize
        }
        
        go p.processHighPriorityBatch(batchSize)
        return
    }
    
    // Only process standard queue if no high-priority items waiting
    standardSize := p.StandardQueue.Size()
    standardAge := p.StandardQueue.OldestItemAge()
    
    if standardSize > 0 && 
       (standardSize >= p.StandardConfig.MinBatchSize || 
        standardAge >= p.StandardConfig.MaxWaitTime) {
        
        // Use dynamic batch sizing based on search traffic
        batchSize := p.getCurrentBatchSize()
        if standardSize < batchSize {
            batchSize = standardSize
        }
        
        go p.processStandardBatch(batchSize)
        return
    }
}

## Implementation Strategy

Based on our benchmark results, here's the recommended implementation approach:

### 1. Model and Configuration

```yaml
embedding:
  model: "nomic-embed-text"  # Clear performance leader
  ollama_num_parallel: 1     # Critical for consistency
  
  high_priority_queue:       # For search queries
    batch_size: 64           # Optimal for throughput with short text
    max_wait_time: 50ms      # Short timeout for interactive use
    min_batch_items: 8       # Process small batches quickly
    max_latency: 500ms       # Target latency for search queries
    
  standard_queue:            # For discussion content
    batch_size: 80           # Maximizes throughput for longer content
    max_wait_time: 200ms     # Longer timeout to build optimal batches
    min_batch_items: 16      # Wait for larger batches
    max_latency: 1000ms      # Acceptable latency for background processing
```

### 2. Service Architecture

Implement a dual-queue processor with the following components:

```go
// EmbeddingQueueManager handles the dual-queue system
type EmbeddingQueueManager struct {
    // Queue configuration
    HighPriorityConfig QueueConfig
    StandardConfig     QueueConfig
    
    // Queues
    highPriorityQueue  *Queue
    standardQueue      *Queue
    
    // Processing state
    processingMutex    sync.Mutex
    processingState    ProcessingState
    
    // Control channels
    shutdownCh         chan struct{}
    
    // Ollama client
    ollamaClient       OllamaClient
    
    // Metrics
    metrics            EmbeddingMetrics
}

// QueueConfig defines settings for each priority queue
type QueueConfig struct {
    MaxBatchSize       int
    MinBatchSize       int           // Minimum items to process a batch before max wait time
    MaxWaitTime        time.Duration // Maximum time to wait for batch formation
    DynamicBatchSizing bool          // Whether to adjust batch size based on load
}

// Queue represents a single priority queue
type Queue struct {
    items              []EmbeddingRequest
    mutex              sync.Mutex
    config             QueueConfig
    lastProcessTime    time.Time
    
    // Statistics
    totalProcessed     int64
    totalLatency       time.Duration
    batchesSent        int64
    batchSizeDistribution map[int]int
    requestsPerMinute  float64 // Rolling average of incoming request rate
}

// ProcessingState tracks the current embedding operation
type ProcessingState struct {
    isProcessing       bool
    currentBatchSize   int
    currentPriority    Priority
    startTime          time.Time
    batchItems         []EmbeddingRequest  // Keep references to items being processed
}

// Queue helper methods
func (q *Queue) Size() int {
    q.mutex.Lock()
    defer q.mutex.Unlock()
    return len(q.items)
}

func (q *Queue) OldestItemAge() time.Duration {
    q.mutex.Lock()
    defer q.mutex.Unlock()
    
    if len(q.items) == 0 {
        return 0
    }
    
    return time.Since(q.items[0].Timestamp)
}

func (q *Queue) Add(req EmbeddingRequest) {
    q.mutex.Lock()
    defer q.mutex.Unlock()
    
    q.items = append(q.items, req)
    
    // Update request rate metrics
    now := time.Now()
    if !q.lastProcessTime.IsZero() {
        interval := now.Sub(q.lastProcessTime).Minutes()
        if interval > 0 {
            // Apply exponential smoothing to the rate calculation
            alpha := 0.3 // Smoothing factor
            newRate := 1.0 / interval // One request in this interval
            q.requestsPerMinute = alpha*newRate + (1-alpha)*q.requestsPerMinute
        }
    }
    q.lastProcessTime = now
}
```

### 3. Processing Logic

Implement the following core processing logic:

```go
// Submit adds a new embedding request to the appropriate queue
func (m *EmbeddingQueueManager) Submit(ctx context.Context, text string, priority Priority) (*EmbeddingResult, error) {
    req := &EmbeddingRequest{
        Text:      text,
        Priority:  priority,
        Timestamp: time.Now(),
        ResultCh:  make(chan *EmbeddingResult, 1),
        Ctx:       ctx,
    }
    
    // Log request details
    logger.With(
        zap.Int("textLength", len(text)),
        zap.String("priority", string(priority)),
        zap.String("requestID", req.ID),
    ).Debug("Received embedding request")
    
    // Add to appropriate queue
    if priority == HighPriority {
        m.highPriorityQueue.Add(req)
        
        // We can't interrupt in-progress batches, but we can log that a high-priority
        // request is waiting behind a standard batch
        if m.processingState.isProcessing && 
           m.processingState.currentPriority == StandardPriority {
            logger.With(
                zap.String("requestID", req.ID),
                zap.Duration("batchRunningTime", time.Since(m.processingState.startTime)),
            ).Info("High priority request waiting for current batch to complete")
            
            // Update metrics for monitoring
            m.metrics.HighPriorityWaiting.Inc()
        }
    } else {
        m.standardQueue.Add(req)
    }
    
    // Wait for result or timeout
    select {
    case result := <-req.ResultCh:
        return result, nil
    case <-ctx.Done():
        return nil, ctx.Err()
    }
}

// startProcessingLoop begins the continuous queue processing
func (m *EmbeddingQueueManager) startProcessingLoop() {
    // Create a ticker that triggers batch processing check
    ticker := time.NewTicker(10 * time.Millisecond)
    defer ticker.Stop()
    
    for {
        select {
        case <-ticker.C:
            m.processNextBatch()
        case <-m.shutdownCh:
            return
        }
    }
}

// processNextBatch decides which queue to process next
func (m *EmbeddingQueueManager) processNextBatch() {
    m.processingMutex.Lock()
    defer m.processingMutex.Unlock()
    
    // If already processing, don't start another batch
    if m.processingState.isProcessing {
        return
    }
    
    // Always check high-priority queue first
    highPrioritySize := m.highPriorityQueue.Size()
    highPriorityAge := m.highPriorityQueue.OldestItemAge()
    
    // Process high priority queue if it has items and meets criteria
    if highPrioritySize > 0 && 
       (highPrioritySize >= m.HighPriorityConfig.MinBatchSize || 
        highPriorityAge >= m.HighPriorityConfig.MaxWaitTime) {
        go m.processHighPriorityBatch()
        return
    }
    
    // Only process standard queue if no high-priority items waiting
    standardSize := m.standardQueue.Size()
    standardAge := m.standardQueue.OldestItemAge()
    
    if standardSize > 0 && 
       (standardSize >= m.StandardConfig.MinBatchSize || 
        standardAge >= m.StandardConfig.MaxWaitTime) {
        go m.processStandardBatch()
        return
    }
}
```

### 4. Batch Processing Implementation

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
   - Embedding vector quality/consistency metrics

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

## Scaling Considerations

Based on our benchmark results, here are concrete recommendations for scaling the embedding service:

### 1. Vertical vs. Horizontal Scaling

For Ollama embedding with `nomic-embed-text`, horizontal scaling is strongly preferred:

- **Each Ollama instance should run with `OLLAMA_NUM_PARALLEL=1`**
- Scale by adding more Ollama instances rather than increasing parallelism
- Consider GPU instances for additional throughput if necessary

### 2. Resource Requirements

Resource allocations based on benchmark results:

| Throughput Target | Resource Allocation |
|-------------------|---------------------|
| ~100 chunks/sec | 1 Ollama instance with 4 CPU cores, 8GB RAM |
| ~200 chunks/sec | 2 Ollama instances each with 4 CPU cores, 8GB RAM |
| ~500 chunks/sec | 5 Ollama instances each with 4 CPU cores, 8GB RAM |

### 3. Auto-Scaling Policy

Implement auto-scaling based on queue depths rather than CPU utilization:

```yaml
autoscaling:
  metricType: Custom
  metricName: nostr_ai_embedding_queue_depth{priority="standard"}
  targetValue: 100
  minReplicas: 2
  maxReplicas: 10
  scaleUpThreshold: 150
  scaleDownThreshold: 50
  stabilizationWindowSeconds: 300
```

## Deployment Configuration

Here's a reference OpenShift deployment configuration that implements the recommendations:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nostr-ai-embedding
  labels:
    app: nostr-ai
    component: embedding
spec:
  replicas: 2
  selector:
    matchLabels:
      app: nostr-ai
      component: embedding
  template:
    metadata:
      labels:
        app: nostr-ai
        component: embedding
    spec:
      containers:
      - name: ollama
        image: ollama/ollama:latest
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        env:
        - name: OLLAMA_NUM_PARALLEL
          value: "1"
        volumeMounts:
        - name: ollama-models
          mountPath: /root/.ollama
        readinessProbe:
          httpGet:
            path: /api/health
            port: 11434
          initialDelaySeconds: 30
          periodSeconds: 10
          
      - name: nostr-ai
        image: bitiq/nostr-ai:latest
        resources:
          requests:
            memory: "1Gi"
            cpu: "0.5"
          limits:
            memory: "2Gi"
            cpu: "1"
        env:
        - name: EMBEDDING_MODEL
          value: "nomic-embed-text"
        - name: OLLAMA_HOST
          value: "localhost:11434"
        - name: HIGH_PRIORITY_BATCH_SIZE
          value: "64"
        - name: HIGH_PRIORITY_MAX_WAIT_MS
          value: "50"
        - name: STANDARD_BATCH_SIZE
          value: "80"
        - name: STANDARD_MAX_WAIT_MS
          value: "200"
        livenessProbe:
          httpGet:
            path: /healthz
            port: 8080
          initialDelaySeconds: 15
          periodSeconds: 20
      
      volumes:
      - name: ollama-models
        persistentVolumeClaim:
          claimName: ollama-models-pvc
```

## Recommended Next Steps

1. **Update Service Configuration**:
   - Switch to nomic-embed-text model for all embedding operations
   - Configure batch sizes of 64 for search queries, 80 for discussion content
   - Set OLLAMA_NUM_PARALLEL=1 for all Ollama instances

2. **Implement Dual-Queue System**:
   - Create separate high-priority and standard queues
   - Add preemption logic for search queries
   - Implement appropriate timeout controls for each queue

3. **Add Comprehensive Monitoring**:
   - Deploy the suggested Prometheus metrics
   - Set up Grafana dashboards to visualize queue performance
   - Implement alerting for queue depth and latency issues

4. **Tune Horizontally**:
   - Scale by adding more Ollama instances rather than adjusting parameters
   - Monitor resource usage to inform better scaling policies
   - Benchmark production workloads to fine-tune batch size parameters

5. **Performance Validation**:
   - Conduct A/B testing with the new configuration
   - Measure end-user perceived latency for search queries
   - Validate that priority system effectively expedites search queries

By implementing these recommendations, the BitIQ nostr_ai service should achieve optimal performance for both search queries and discussion content processing, with search queries receiving the priority treatment required for excellent user experience.