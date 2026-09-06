# Part B — Capacity Reconciliation

## Model & Hardware Reference
From `bench/model_spec.md`:
- **Model:** FLM-4B-Instruct (dense, 4.2B params)
- **GPU:** 1× NVIDIA L4 (24 GB VRAM, 300 GB/s bandwidth, ~121 TFLOPS fp16)
- **Layers:** 28 | **KV heads (GQA):** 8 | **head_dim:** 128
- **KV cache precision:** fp16 (2 bytes/element)
- **`gpu_memory_utilization`:** 0.92
- **Non-KV overhead:** ~1.6 GB

---

## B1 — KV-Cache Bytes per Token + Max Concurrent Sequences (7 pts)

### (a) KV-cache bytes per token — exact calculation

Each transformer layer stores a **Key** and **Value** tensor for every token.
With Grouped Query Attention (GQA), KV heads = 8 (not 24).

```
KV bytes per token per layer = 2 (K + V) × kv_heads × head_dim × bytes_per_element
                              = 2 × 8 × 128 × 2
                              = 4,096 bytes per layer

KV bytes per token (all layers) = 4,096 × 28 layers
                                 = 114,688 bytes
                                 = 112 KB per token
```

**Answer: 114,688 bytes (112 KB) per token.**

### (b) Maximum concurrent 4096-token sequences

**Step 1: Available VRAM after weights**
```
Total GPU memory          = 24 GB = 24 × 1024³ bytes = 25,769,803,776 bytes
Usable (× 0.92)          = 25,769,803,776 × 0.92 = 23,708,499,435 bytes = ~22.08 GB
Model weights (fp16)      = 4.2 × 10⁹ × 2 = 8,400,000,000 bytes = ~7.83 GB
Non-KV overhead           = 1.6 GB = 1,717,986,918 bytes
```

**Step 2: Memory available for KV cache**
```
KV budget = 22.08 GB - 7.83 GB - 1.60 GB
           = 12.65 GB = 13,581,614,080 bytes
```

**Step 3: Total tokens that fit in KV cache**
```
Max tokens = 13,581,614,080 / 114,688 = ~118,430 tokens
```

**Step 4: Max concurrent 4096-token sequences**
```
Max sequences = 118,430 / 4,096 = 28.9 → floor to 28 sequences
```

**Answer: ~28 concurrent sequences at max_model_len = 4096.**

### Verification against bench_log.csv

| batch | prompt | gen | total_per_seq | kv_util | expected_seqs_in_cache |
|-------|--------|-----|--------------|---------|------------------------|
| 24    | 3584   | 512 | 4096         | 0.93    | 24 × 4096 = 98,304 tok |
| 32    | 3584   | 512 | 4096         | 0.97    | 32 × 4096 = 131,072 tok |

At batch 24, KV util = 0.93:
- Predicted KV capacity: 118,430 tokens × 0.93 ≈ 110,140 tokens
- Actual used (24 seqs × 4096 tok): 98,304 tokens → 83% of capacity
- At batch 32: 131,072 tokens would require 131,072/118,430 = **110.7% of capacity** → overflows → preemption

The log confirms preemption starts at batch 32 (`preempted_seqs = 7`), which aligns with our prediction of ~28 max sequences.

**Discrepancy check:** kv_cache_util = 0.97 at batch 32 means the vLLM KV block allocator reports 97% of *allocatable* KV blocks are used. The allocatable block count is determined by the actual runtime, not just our static estimate, but the order of magnitude matches: ~28 sequences at 4096 tokens each.

---

## B2 — Throughput Anomaly in Long-Context Sweep (6 pts)

### Identification

From `bench_log.csv`, the long-context sweep (prompt_len = 3584, gen_len = 512):

| batch | reported_tok_s | kv_cache_util | preempted_seqs |
|-------|---------------|---------------|----------------|
| 4     | 565.4         | 0.16          | 0              |
| 8     | 902.6         | 0.31          | 0              |
| 16    | 1,311.4       | 0.62          | 0              |
| **24**| **1,607.4**   | **0.93**      | **0**          |
| 32    | 1,384.0       | 0.97          | **7**          |
| 48    | 1,298.5       | 0.97          | **23**         |

**Anomaly:** Throughput peaks at batch 24 (1,607 tok/s) then **falls** at batch 32 (1,384 tok/s) and continues falling at batch 48 (1,299 tok/s), despite having more requests in flight.

### Mechanism

Naively, doubling batch size doubles GPU utilization and throughput. But here:

1. **KV cache exhaustion triggers preemption.** At batch 32, total token demand = 32 × 4096 = 131,072 tokens, exceeding the ~118K token KV capacity. vLLM's scheduler must **preempt** (swap out or recompute) 7 sequences. Preempted sequences have their KV cache evicted and must be **re-prefilled** when rescheduled — wasting GPU compute on redundant work.

2. **kv_cache_util saturates at 0.97** (not 1.0 because of block granularity). Once saturated, adding more requests doesn't increase throughput — it just increases preemption overhead.

3. **wall_clock grows faster than n_requests.** At batch 32: 32 requests / 94.71s = 0.338 requests/s. At batch 24: 24 / 61.16 = 0.392 requests/s. Per-request throughput is actually *worse* at batch 32.

### Evidence (specific rows and columns)
- **batch 24 → batch 32:** `kv_cache_util` jumps from 0.93 → 0.97; `preempted_seqs` goes 0 → 7; `reported_tok_s` drops 1,607 → 1,384 (-14%).
- **batch 32 → batch 48:** `preempted_seqs` rises 7 → 23; `reported_tok_s` drops further to 1,299.
- **ttft_ms_p50** at batch 48 = 955 ms (nearly 2× the batch 16 value of 498 ms), confirming decode is waiting on preemption/recomputation.

### Proposed Fix + Predicted Effect

**Config change:** Set `max_num_seqs = 24` (limit concurrent sequences to stay below KV saturation point) combined with **chunked prefill** (`enable_chunked_prefill = True`) to keep GPU compute utilized.

**Predicted quantitative effect:**
- Capping at 24 sequences eliminates preemption → sustained 1,607 tok/s instead of degraded 1,299–1,384 tok/s
- With chunked prefill, new requests can begin prefill in parallel with decode, reducing TTFT without starving KV cache

**Alternatively:** Reduce `max_model_len` from 4096 to 3072 (removing unused buffer) to fit ~38 sequences at full capacity.

---

## B3 — REPORT_v0 Misreading Analysis (4 pts)

### The claim

> REPORT_v0, Section 2: *"at batch 16, long prompts hit 1311 tok/s vs only 883 tok/s for short prompts. Longer prompts clearly give better GPU utilization."*
> *"assume ~1600 tok/s per L4 (best observed) and scale linearly with batch size, so batch 48 should give us ~3200 tok/s."*

### The misreading

Both conclusions come from misinterpreting the `reported_tok_s` column.

**`reported_tok_s` = (total input tokens + generated tokens) / wall_clock_s**

It counts *all* token processing — prefill (prompt tokens) AND decode (generated tokens). When prompts are longer, there are more prefill tokens counted in the numerator, making the metric look higher even if *generated* throughput is unchanged or worse.

**For short prompts (batch 16):**
- `prompt_len = 512`, `gen_len = 256`, total tokens per seq = 768
- `reported_tok_s = 883.2`
- Total tokens processed = 16 × 768 = 12,288
- wall_clock = 13.91s → 12,288/13.91 = **883 ✓**

**For long prompts (batch 16):**
- `prompt_len = 3584`, `gen_len = 512`, total per seq = 4096
- `reported_tok_s = 1311.4`
- Total tokens = 16 × 4096 = 65,536
- wall_clock = 49.97s → 65,536/49.97 = **1311 ✓**

The metric is computing a different *thing* for each row. Comparing them directly is like comparing apples and oranges.

### What the honest "goodput" is for batch 24 long-prompt

**Goodput = generation throughput** (only tokens the *user receives*, not prompt overhead).

**Method 1 — generation tokens only:**
```
goodput = num_requests × gen_len / wall_clock_s
        = 24 × 512 / 61.16
        = 12,288 / 61.16
        = 200.9 tok/s (generated tokens)
```

**Method 2 — from itl_ms_p50 (inter-token latency):**
```
# At median ITL, each sequence generates 1 token every itl_ms_p50 ms
# 24 concurrent sequences at 96.07 ms ITL:
goodput ≈ 24 × (1000 / 96.07) = 24 × 10.41 = 249.9 tok/s
```

Both methods give ~200–250 tok/s of *user-facing* generation throughput, not 1,607 tok/s.

### What the report should have said

> *"At batch 24 with 3584-token prompts, the serving stack processes 1,607 total tok/s, but this includes 3584/4096 = 87.5% prompt tokens. Honest user-facing generation throughput is only ~200 tok/s. For capacity planning, distinguish between prefill throughput (affects latency) and decode throughput (affects user-perceived token speed). Batch 48 does NOT deliver 3200 tok/s — it degrades to 1,299 tok/s due to KV cache preemption."*

---

## B4 — Monitoring Metric to Confirm B2 Mechanism (3 pts)

**Metric to pull:**

> **`num_preemptions_total`** (vLLM Prometheus metric) — the cumulative count of sequences preempted by the KV cache scheduler.

Alternatively: `gpu_cache_usage_perc` (KV cache block utilization) and `num_running_seqs`.

**What to expect:**
- At `max_num_seqs ≤ 24` with the current setup: `num_preemptions_total` should remain **0** across the entire load test.
- At `max_num_seqs = 32`: `num_preemptions_total` should spike to ~7+ per run, and `gpu_cache_usage_perc` should reach ~97% and plateau.
- Correlation: `num_preemptions_total > 0` ↔ `reported_tok_s` drops below the batch 24 peak → confirms the preemption-throughput link.

**Secondary confirmation:** `ttft_ms_p50` should jump non-linearly (batch 24→32: 500ms→637ms, a 27% increase for a 33% increase in batch size) because preempted sequences are waiting for KV space before their prefill can complete.
