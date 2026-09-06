# flamAI AI Assignment 2026 — The Audit

This repository contains the comprehensive technical audit of `REPORT_v0.md`, `fertility.py`, and `bench_log.csv` as specified in `AI ASSIGNMENT 2026.pdf`.

---

## 🔍 Key Findings & Executive Summary

### Part A — Tokenizer Audit (50 pts)
1. **Flaws Isolated in `fertility.py` (The Evidence Rule):**
   - **Bug 1 (Code Bug):** `split(" ")` on line 62 creates phantom empty strings on multiple spaces (`"books  in"`), **underestimating fertility** by inflating word counts (+0.15 tok/word delta on Hindi).
   - **Bug 2 (Conceptual Bug):** `.lower()` on line 59 artificially splits uppercase acronyms (`NASA` $\rightarrow$ 1 token, `nasa` $\rightarrow$ 2 tokens), **overestimating English fertility** and falsely compressing the cross-lingual ratio.
   - **Bug 3 (Metric Aggregation Bug):** Equal-weight line averaging (`sum(per_line)/n`) distorts corpus-level fertility relative to true aggregate (`total_tokens / total_words`).
   - **Harmless Code:** `random.seed(1337)` on line 25 is never called; verified $\Delta = 0.000$ (zero effect).
2. **Corrected Multilingual & Multi-Denominator Analysis:**
   - On `gpt2`, Hindi is **$6.38\times$ worse** and Dravidian languages (Kannada, Tamil) are **$10.5\times\text{--}11.4\times$ worse** per sentence because `tok/byte \approx 0.98\text{--}0.99` (GPT-2 has zero Dravidian vocabulary merges and emits raw individual UTF-8 bytes).
   - On Google's Indic-aware **MuRIL**, Hindi drops to **$0.89\times$** and Kannada/Tamil drop to **$0.74\text{--}0.76\times$** of English per sentence.
   - **Core Conclusion:** High Indic fertility is an artifact of tokenizer vocabulary bias, not an inherent property of Indic scripts.
   - **Single Routing Metric:** **Tokens per Semantic Unit (Sentence / Request Turn)** holds information meaning constant across languages.

### Part B — Capacity Reconciliation (20 pts)
1. **Exact KV-Cache Footprint (B1):**
   $$\text{Bytes/token} = 2 \times 28 \text{ layers} \times 8 \text{ KV heads} \times 128 \text{ dim} \times 2 \text{ bytes (fp16)} = 114,688 \text{ bytes} = \mathbf{112\text{ KB/token}}$$
   - Usable KV VRAM on 24GB L4 GPU $= 12.08\text{ GB} \implies \mathbf{27\text{--}28\text{ max concurrent 4096-token sequences}}$.
2. **Throughput Anomaly & Preemption (B2):**
   - At batch $\ge 32$ with prompt 3584, memory demand ($131,072\text{ tok}$) exceeds KV capacity ($113\text{k tok}$), causing vLLM to trigger **7 preemptions at batch 32** and **23 at batch 48**. Throughput collapses from peak **$1607\text{ tok/s} \rightarrow 1384 \rightarrow 1298\text{ tok/s}$**.
   - **Fix:** Cap `max_num_seqs = 24` with chunked prefill enabled (`enable_chunked_prefill = True`).
3. **Report Misreading & Honest Goodput (B3):**
   - `REPORT_v0` falsely claimed long prompts improve GPU throughput and extrapolated 3200 tok/s at batch 48.
   - `reported_tok_s` conflated prompt prefill tokens (87.5% of total) with generation.
   - **Honest Batch-24 Decode Goodput:** $\frac{24 \times 512}{61.16\text{ s}} = \mathbf{200.9\text{ tok/s}}$ (or $249.8\text{ tok/s}$ via median ITL).
4. **Monitoring Counter (B4):** `vllm:num_preemptions_total` in Prometheus (must stay $0$).

### Part C — Indic Casual Tone Strategy (15 pts)
- **Recommendation:** **Option (a) SFT with LoRA on synthetic pairs**, launching with reviewer-validated Hindi + Kannada first.
- **Arithmetic:** 12k synthetic pairs ($\approx 24\text{h}$ on A100), 18h LoRA training, 20h reviewer time validating 2,400 samples.
- **Success Metric:** $\ge 70\%$ of responses rated casual on a 100-query held-out set by native reviewer.
- **Kill Criterion:** Abort by Day 12 if Hindi human eval shows $< 50\%$ casual rating or COMET drops $> 3$ points.
- **Day-1 Experiment:** 50-sample zero-shot prompt test to measure prompt-only baseline ceiling.

---

## 📂 Deliverables Directory Structure

All audit deliverables are located in [`your-submission/`](your-submission/):

```
your-submission/
├── NOTEBOOK.md                 # Chronological lab notebook with hypotheses, experiments & revisions
├── AI_USAGE.md                 # Honest summary of AI assistance & hallucination disclosure
├── partA/
│   ├── corpus/                 # 4-language evaluation corpus (eng, hin, kan, tam + metadata)
│   ├── scripts/
│   │   ├── build_corpus.py     # Multilingual corpus builder
│   │   ├── audit_bugs.py       # Bug isolation experiments with measured evidence
│   │   └── fertility_fixed.py  # Production-ready corrected fertility calculation script
│   ├── audit_bug_results.txt   # Measured before/after evidence output
│   ├── results_gpt2.txt        # GPT-2 benchmark results across all 4 denominators
│   ├── results_muril.txt       # Indic-aware MuRIL benchmark results
│   ├── routing_memo.md         # A4 ≤1 page Leadership Routing Memo
│   └── PART_A.md               # Part A detailed audit report
├── partB/
│   └── PART_B.md               # Part B capacity reconciliation & KV-cache arithmetic
└── partC/
    └── memo.md                 # Part C casual Indic adaptation decision memo
```

---

## 🛠️ Quick Verification Commands

```bash
# 1. Run controlled bug isolation experiment (Part A2)
python your-submission/partA/scripts/audit_bugs.py --eng starter_kit/corpus_sample/eng_sample.txt --hin starter_kit/corpus_sample/hin_sample.txt

# 2. Run corrected fertility analysis on multilingual corpus (Part A3)
python your-submission/partA/scripts/fertility_fixed.py --corpus eng=your-submission/partA/corpus/eng.txt --corpus hin=your-submission/partA/corpus/hin.txt --corpus kan=your-submission/partA/corpus/kan.txt --corpus tam=your-submission/partA/corpus/tam.txt --tokenizer gpt2 --compare-buggy
```
