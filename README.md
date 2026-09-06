# flamAI AI Assignment 2026 — The Technical Audit

[![Audit Status](https://img.shields.io/badge/Audit-Verified%20100%25-brightgreen)](your-submission/)
[![Defense Ready](https://img.shields.io/badge/Defense-Prepared-blue)](your-submission/NOTEBOOK.md)
[![Desmos Integrated](https://img.shields.io/badge/Desmos-Interactive%20Visualizer-orange)](your-submission/desmos_visualizer.html)

This repository contains the rigorous technical audit of `REPORT_v0.md`, `fertility.py`, and `bench_log.csv` as specified in `AI ASSIGNMENT 2026.pdf`.

---

## 📊 Executive Summary: REPORT_v0 vs. Corrected Audit

| Dimension | REPORT_v0 (Previous Intern) | Corrected Audit (Our Findings) | Impact & Decision Shift |
|:---|:---|:---|:---|
| **Hindi Fertility vs English** | **5.89× worse** (reported as script property) | **6.38× worse on GPT-2**, but **0.89× on MuRIL** | High fertility is **tokenizer vocabulary bias**, not script nature. |
| **Dravidian Languages (Kan/Tam)** | *Unmeasured / Ignored* | **10.5×–11.4× worse on GPT-2** (`tok/byte ≈ 0.99`) | GPT-2 lacks Dravidian merges and emits raw bytes. Solved by Indic tokenizers. |
| **Serving Cost Recommendation** | Budget 6× serving cost for Indic traffic | **Deploy Indic-aware vocabulary tokenizer** | Eliminates the 6× penalty, bringing Indic inference cost to **~1.0×–1.3×** of English. |
| **Long-Prompt GPU Throughput** | Claimed longer prompts give better throughput | **Throughput collapses at batch ≥ 32** due to preemption | Preemption thrashing at batch 32 (7 preempts) and batch 48 (23 preempts). |
| **Batch 48 Throughput Claim** | Fictional prediction of **~3,200 tok/s** | Actually **1,298.5 tok/s** (a 19% drop from peak) | Naive linear extrapolation failed to model KV-cache exhaustion. |
| **Batch 24 Honest Goodput** | Reported **1,607.4 tok/s** (prefill + decode) | Honest decode goodput is **200.9 tok/s** | 87.5% of reported tokens were prompt prefill overhead. |

---

## 🔬 Part A: The Tokenizer Audit

### 1. Bug Isolation & The Evidence Rule (`fertility.py`)

All flaws in `fertility.py` were isolated, measured, and verified on the starter corpus:

| Bug / Code Pattern | Classification | Line in `fertility.py` | Measured Before & After Numbers | Distortion & Direction |
|:---|:---|:---:|:---|:---|
| **`split(" ")` vs `split()`** | Code Bug | [L62](starter_kit/fertility.py#L62) | • Buggy Hindi: `7.4485`<br>• Fixed Hindi: `7.5985`<br>• **Delta: +0.1500** | **Underestimates fertility** by counting phantom empty strings from double spaces (`"books  in"`) as words. |
| **`.lower()` Preprocessing** | Conceptual Bug | [L59](starter_kit/fertility.py#L59) | • Lowercased Eng: `1.2831`<br>• Original Eng: `1.2472`<br>• **Delta: -0.0359** (Hindi $\Delta = 0$) | **Overestimates English tokens** by splitting acronyms (`NASA` $\rightarrow$ 1 tok, `nasa` $\rightarrow$ 2 tok), artificially compressing the ratio. |
| **Mean-of-Ratios Aggregation** | Aggregation Bug | [L67](starter_kit/fertility.py#L67) | • Mean-of-ratios Hin: `7.5985`<br>• Corpus total/total Hin: `7.5246`<br>• **Delta: -0.0739** | **Biases results toward short sentences** by weighting a 3-word sentence equally with a 30-word sentence. |
| **`random.seed(1337)`** | Harmless Code | [L25](starter_kit/fertility.py#L25) | • Measured Delta: **0.0000** | **Zero effect.** `random` is never called; tokenization is deterministic. |

---

### 2. Multilingual & Multi-Denominator Evaluation

Measured across 4 languages on our evaluation corpus comparing **GPT-2** (English-centric) against **Google MuRIL** (Indic-specialized):

| Language | Script Family | Tokenizer | Tokens / Word | Tokens / Grapheme | Tokens / UTF-8 Byte | Tokens / Sentence | Cost Ratio vs English |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **English (`eng`)** | Latin (Germanic) | `gpt2`<br>`muril` | 1.280<br>1.269 | 0.206<br>0.204 | 0.206<br>0.204 | 26.8<br>26.6 | **1.00× (Baseline)**<br>**1.00× (Baseline)** |
| **Hindi (`hin`)** | Devanagari (Indo-Aryan) | `gpt2`<br>`muril` | 8.201<br>1.136 | 2.420<br>0.335 | 0.593<br>0.082 | 171.4<br>23.7 | **6.38× worse**<br>**0.89× (Parity!)** |
| **Kannada (`kan`)** | Kannada (Dravidian) | `gpt2`<br>`muril` | 23.255<br>1.672 | 3.999<br>0.287 | **0.980**<br>0.070 | 281.1<br>20.2 | **10.47× worse**<br>**0.76× (Parity!)** |
| **Tamil (`tam`)** | Tamil (Dravidian) | `gpt2`<br>`muril` | 23.380<br>1.514 | 4.108<br>0.266 | **0.994**<br>0.064 | 305.5<br>19.8 | **11.38× worse**<br>**0.74× (Parity!)** |

> **Key Takeaway:** The only denominator that holds information meaning constant across languages is **Tokens per Semantic Unit (Sentence / Request Turn)**. "Words" fail due to Dravidian agglutinative morphology, and bytes hold data volume rather than meaning constant.

---

## ⚡ Part B: Capacity Reconciliation & Serving Physics

### 1. Hardware & Model Parameter Specifications

| Parameter | Value | Mathematical Derivation / Role |
|:---|:---|:---|
| **Base Model** | FLM-4B-Instruct | 4.2 Billion parameters, dense transformer |
| **Layer Count ($L$)** | 28 layers | Stored in transformer stack |
| **Query Heads ($Q$) / KV Heads ($KV$)** | 24 Q Heads / 8 KV Heads | Grouped Query Attention (GQA factor = 3) |
| **Head Dimension ($d$)** | 128 | Vector dimension per attention head |
| **KV Precision ($P$)** | fp16 (2 bytes) | 2 bytes per scalar value |
| **GPU Hardware** | 1× NVIDIA L4 (24 GB VRAM) | 300 GB/s peak memory bandwidth, ~121 TFLOPS fp16 |
| **Usable VRAM Ceiling** | **22.08 GB** | $24\text{ GB} \times 0.92\text{ gpu\_memory\_utilization}$ |
| **Static Weight Footprint** | **8.40 GB** | $4.2 \times 10^9 \text{ params} \times 2\text{ bytes (fp16)}$ |
| **Runtime Overhead (CUDA/Activations)** | **1.60 GB** | Activations, CUDA graphs, working buffers |
| **VRAM Budget for KV Cache** | **12.08 GB** | $22.08\text{ GB} - 8.40\text{ GB} - 1.60\text{ GB} = 12,970,803,200\text{ bytes}$ |

$$\text{KV-Cache Bytes per Token} = 2 \times L \times KV \times d \times P = 2 \times 28 \times 8 \times 128 \times 2 = \mathbf{114,688\text{ bytes}} = \mathbf{112\text{ KB/token}}$$

$$\text{Max KV Token Capacity} = \frac{12,970,803,200\text{ bytes}}{114,688\text{ bytes/token}} = \mathbf{113,097\text{ tokens}}$$

$$\text{Max Concurrent 4096-token Sequences} = \frac{113,097\text{ tokens}}{4,096\text{ tokens/seq}} = \mathbf{27.6} \implies \mathbf{27\text{--}28\text{ sequences}}$$

---

### 2. Log Analysis & Anomaly Breakdown (`bench_log.csv`)

| Batch Size | Prompt / Gen Len | Total Tokens / Req | Wall Clock (s) | Reported Throughput | KV Cache Util | Preempted Seqs | Median TTFT / ITL | Honest Decode Goodput |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **4** | 3584 / 512 | 4096 | 28.98 s | 565.4 tok/s | 0.16 | 0 | 483.2 ms / 51.3 ms | 70.7 tok/s |
| **8** | 3584 / 512 | 4096 | 36.30 s | 902.6 tok/s | 0.31 | 0 | 519.0 ms / 62.3 ms | 112.8 tok/s |
| **16** | 3584 / 512 | 4096 | 49.97 s | 1,311.4 tok/s | 0.62 | 0 | 498.3 ms / 77.2 ms | 163.9 tok/s |
| **24** | **3584 / 512** | **4096** | **61.16 s** | **1,607.4 tok/s (Peak)** | **0.93** | **0** | **500.5 ms / 96.1 ms** | **200.9 tok/s** |
| **32** | 3584 / 512 | 4096 | 94.71 s | **1,384.0 tok/s (Drop)** | **0.97** | **7 ⚠️** | 636.9 ms / 101.8 ms | 172.9 tok/s |
| **48** | 3584 / 512 | 4096 | 151.41 s | **1,298.5 tok/s (Drop)** | **0.97** | **23 ⚠️** | 955.4 ms / 100.0 ms | 162.4 tok/s |

> **Preemption Mechanism & Solution:** 
> - At batch 32, required tokens ($131,072$) exceed physical KV capacity ($113,097$). vLLM is forced to preempt sequences, incurring redundant prefill recomputation.
> - **Production Fix:** Set `max_num_seqs = 24` with chunked prefill enabled (`enable_chunked_prefill = True`).
> - **Monitoring Metric:** Monitor `vllm:num_preemptions_total` in Prometheus (must remain $0$).

---

## 🎯 Part C: Decision Memo — Indic Casual Tone Adaptation

Comparison of paths for casual conversational tone across 6 Indic languages (Hindi, Kannada, Tamil, Telugu, Bengali, Marathi) under constraints: **1× A100-80GB (2 weeks), 1 Reviewer (Hindi + Kannada, 10h/wk), 3 weeks to launch**.

| Strategic Option | Feasibility on A100 | Serving Latency / Cost Impact | Quality & Consistency | Reviewer Coverage Fit | Recommendation & Action |
|:---|:---|:---|:---|:---|:---|
| **(a) SFT with LoRA on Synthetic Pairs** | **High** (18h training on 4B model) | **Zero added latency** (fp16 adapter loading) | **High** (register learned directly in weights) | Validates 2,400 pairs across Hindi + Kannada | **PRIMARY CHOICE** (Ship Hindi + Kannada at launch; stage remaining 4). |
| **(b) Small (≤1B) Rewriter Model** | Medium | **High Penalty** (doubles inference steps & latency) | Medium | Same partial coverage | **REJECTED** (doubling serving cost for Indic is prohibitive). |
| **(c) Prompt Engineering Only** | Very High (0 GPU cost) | Negligible | Low (unreliable style consistency at scale) | Fast baseline test | **DAY-1 BASELINE** (Run 50-sample test to establish baseline ceiling). |

- **Success Metric:** $\ge 70\%$ of responses rated "appropriately casual" ($\le 2$ on 5-point formality scale) on a 100-query held-out set by native reviewer (vs $\le 20\%$ baseline).
- **Kill Criterion:** Abort by Day 12 if Hindi human evaluation shows $< 50\%$ casual rating or COMET score drops $> 3$ points.

---

## 📂 Repository Deliverables Map

```
your-submission/
├── NOTEBOOK.md                 # Chronological lab notebook with hypotheses, experiments & revisions
├── AI_USAGE.md                 # Transparent AI disclosure detailing acceleration vs hallucinations
├── DESMOS_GUIDE.md             # Desmos mathematical formulas, bounds, and graphing instructions
├── desmos_visualizer.html      # Interactive standalone web application with embedded Desmos API
├── partA/
│   ├── corpus/                 # 4-language evaluation corpus (eng, hin, kan, tam + metadata)
│   ├── scripts/
│   │   ├── build_corpus.py     # Multilingual Wikipedia corpus extraction script
│   │   ├── audit_bugs.py       # Bug isolation & evidence verification script
│   │   └── fertility_fixed.py  # Production corrected fertility engine (4 denominators, 2 tokenizers)
│   ├── audit_bug_results.txt   # Verified evidence output
│   ├── results_gpt2.txt        # GPT-2 benchmark results
│   ├── results_muril.txt       # Google MuRIL benchmark results
│   ├── routing_memo.md         # A4 ≤1 page Leadership Routing Memo
│   └── PART_A.md               # Detailed Part A technical audit write-up
├── partB/
│   └── PART_B.md               # Part B capacity reconciliation & mathematical proofs
└── partC/
    └── memo.md                 # Part C casual Indic adaptation decision memo
```

---

## 🚀 Quick Reproduction Commands

```powershell
# 1. Ensure UTF-8 output encoding
$env:PYTHONIOENCODING="utf-8"

# 2. Run bug isolation experiments (Part A2 Evidence Rule)
python your-submission/partA/scripts/audit_bugs.py --eng starter_kit/corpus_sample/eng_sample.txt --hin starter_kit/corpus_sample/hin_sample.txt

# 3. Run corrected fertility analysis on GPT-2 (Part A3)
python your-submission/partA/scripts/fertility_fixed.py --corpus eng=your-submission/partA/corpus/eng.txt --corpus hin=your-submission/partA/corpus/hin.txt --corpus kan=your-submission/partA/corpus/kan.txt --corpus tam=your-submission/partA/corpus/tam.txt --tokenizer gpt2 --compare-buggy

# 4. Run corrected fertility analysis on Google MuRIL (Part A3)
python your-submission/partA/scripts/fertility_fixed.py --corpus eng=your-submission/partA/corpus/eng.txt --corpus hin=your-submission/partA/corpus/hin.txt --corpus kan=your-submission/partA/corpus/kan.txt --corpus tam=your-submission/partA/corpus/tam.txt --tokenizer hf:google/muril-base-cased
```
