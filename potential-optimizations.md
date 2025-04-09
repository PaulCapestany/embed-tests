# Ollama Embedding Optimization for nostr_ai: Multi-Model Analysis and Implementation Strategy

## Executive Summary

This document analyzes multi-model benchmark results for the BitIQ `nostr_ai` service's embedding workflow and provides a concrete implementation strategy based on the findings. Our testing reveals significant performance variations across embedding models, with implications for both system architecture and queue management.

Key findings:
- **nomic-embed-text** significantly outperforms other models in throughput (~122 chunks/s vs 56-59 chunks/s for others)
- Optimal batch sizes vary by text length and model, but generally converge around 64-80
- With `OLLAMA_NUM_PARALLEL=1`, all models maintain perfect embedding consistency
- Different queue management strategies are required for search queries vs. discussion content

These results directly inform our dual-queue implementation strategy for handling high-priority search queries alongside longer discussion content.

## Model Performance Analysis

### Overall Performance Ranking

Based on extensive benchmarking, here's how the models rank:

| Model | Avg Max Throughput | Avg Max Speedup | Optimal for |
|-------|-------------------|----------------|------------|
| nomic-embed-text | 122.33 chunks/s | 2.53x | Search queries, high throughput needs |
| mxbai-embed-large | 59.43 chunks/s | 1.63x | Balanced performance |
| bge-m3 | 56.99 chunks/s | 4.08x | Maximum batch efficiency |
| snowflake-arctic-embed2 | 56.11 chunks/s | 4.07x | Maximum batch efficiency |

### Performance by Text Length

Shorter text (search queries) and longer text (discussion content) show different performance characteristics:

#### Short Queries (15 chars)

| Model | Best Batch Size | Max Throughput | Latency |
|-------|----------------|----------------|---------|
| nomic-embed-text | 64 | ~170 chunks/s | 372ms |
| mxbai-embed-large | 64 | ~97 chunks/s | 662ms |
| snowflake-arctic-embed2 | 80 | ~84 chunks/s | 950ms |
| bge-m3 | 80 | ~87 chunks/s | 914ms |

#### Discussion Content (256 chars)

| Model | Best Batch Size | Max Throughput | Latency |
|-------|----------------|----------------|---------|
| nomic-embed-text | 80 | ~104 chunks/s | 770ms |
| mxbai-embed-large | 64 | ~44 chunks/s | 1445ms |
| snowflake-arctic-embed2 | 80 | ~45 chunks/s | 1762ms |
| bge-m3 | 64 | ~46 chunks/s | 1379ms |

### Key Insights

1. **Model Selection**: 
   - **nomic-embed-text** provides significantly higher throughput and lower latency than alternatives
   - This performance advantage is particularly pronounced for short text (search queries)
   - For the critical search query use case, nomic-embed-text is more than 2x faster than alternatives

2. **Batch Size Optimization**:
   - Short queries (15 chars): Batch size 64 is optimal for nomic-embed-text
   - Discussion content (256+ chars): Batch size 80 is optimal for nomic-embed-text
   - Larger batch sizes increase throughput but add latency - this creates a tradeoff for queue management

3. **Consistency Confirmation**:
   - All models show perfect embedding consistency when `OLLAMA_NUM_PARALLEL=1`
   - This confirms our previous finding that parallelism should be achieved through horizontal scaling rather than increased parallel processing within Ollama instances

## Queue Design Recommendations

Based on these findings, we recommend implementing a prioritized dual-queue architecture that directly addresses the different requirements for search queries and discussion content.

### High-Priority Queue (Search Queries)

- **Purpose**: Process user search queries with minimal latency
- **Implementation**: 
  - Use preemptive queue with ability to interrupt ongoing batches
  - Optimize for low latency over maximum throughput
  - Process immediately when queue depth reaches 8-16 items
  - Maximum wait time of 20-50ms to form batches during low traffic
  - Never wait for batch formation during high traffic

### Standard Queue (Discussion Content)

- **Purpose**: Process longer discussion content with maximum throughput
- **Implementation**:
  - Use batch-oriented queue that optimizes for throughput
  - Larger batch sizes (64-80) for maximum efficiency
  - Wait time of 100-300ms to form optimal batches
  - Can be preempted by high-priority queue

### Queue Management Policy

Since Ollama's API doesn't allow canceling in-progress embedding requests, we need a priority-based queue management system that ensures high-priority items are processed immediately after any currently executing batch completes.

The system should implement the following queue management logic:

```
// This function runs on a short interval (e.g., every 10ms)
function processNextBatch() {
    // If already processing a batch, do nothing and wait for completion
    if (isProcessingAnyBatch) {
        return;
    }
    
    // Always check high-priority queue first
    if (high_priority_queue.length > 0) {
        // Process immediately if we have enough items or oldest has waited too long
        if (high_priority_queue.length >= min_high_priority_batch_size || 
            high_priority_queue.oldestItemAge > max_high_priority_wait_time) {
            process_high_priority_queue();
        }
    } 
    // Only process standard queue if no high-priority items are waiting
    else if (standard_queue.length > 0) {
        // Process if we have enough items or oldest has waited too long
        if (standard_queue.length >= optimal_batch_size || 
            standard_queue.oldestItemAge > max_standard_wait_time) {
            process_standard_queue();
        }
    }
}
```

This approach ensures that high-priority search queries are processed as soon as possible after any currently executing batch completes.

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