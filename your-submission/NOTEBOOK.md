# Chronological Lab Notebook: flamAI Tokenizer & Serving Audit

This notebook records the live, chronological investigation into `REPORT_v0.md`, `fertility.py`, and `bench_log.csv`. Every entry documents a specific **Hypothesis → Experiment / Command → Result / Dead End → Revision / Finding**.

---

## Log Entry 1 — Initial Reconnaissance & Codebase Inspection
- **Timestamp:** 2026-09-06 23:04 IST
- **Hypothesis:** `REPORT_v0.md` claims Hindi fertility is 5.89× worse than English and concludes leadership should budget 6× serving costs and route Indic traffic to a separate model. We suspect both code-level bugs in `fertility.py` and conceptual metric aggregation errors.
- **Action:** Read `REPORT_v0.md`, `fertility.py`, `model_spec.md`, and `bench_log.csv`.
- **Observations:**
  - `fertility.py` uses `line.split(" ")` on line 62.
  - `fertility.py` runs `line = line.lower()` on line 59 before encoding.
  - `fertility.py` computes `sum(per_line_fertility) / n` (mean of ratios) rather than `total_tokens / total_words`.
  - `random.seed(1337)` is set on line 25, but `random` is never called.
  - `bench_log.csv` shows at `prompt_len=3584`, throughput peaks at batch 24 (1607 tok/s) and drops at batch 32 (1384 tok/s) and batch 48 (1298 tok/s) with `preempted_seqs` rising from 0 to 7 to 23.

---

## Log Entry 2 — Part A1: Multilingual Eval Corpus Construction & Dead Ends
- **Timestamp:** 2026-09-06 23:14 IST
- **Hypothesis:** We need a 4-language eval corpus including English, Hindi, and two Dravidian languages (Kannada, Tamil). We initially attempt to download the standard FLORES-200 devtest split (`eng_Latn`, `hin_Deva`, `kan_Knda`, `tam_Taml`).
- **Experiment 1 (Dead End):** 
  ```python
  from datasets import load_dataset
  ds = load_dataset('facebook/flores', 'eng_Latn', split='devtest')
  ```
  - **Result:** `DatasetNotFoundError: Dataset 'facebook/flores' is a gated dataset on the Hub. You must be authenticated to access it.`
- **Experiment 2 (Dead End):**
  - Attempted Hugging Face mirrors (`Muennighoff/flores200`, `openlanguagedata/flores_200`, `bri25yu/flores200_val_test`).
  - **Result:** Gated or raised `RuntimeError: Dataset scripts are no longer supported`.
- **Revision / Working Solution:**
  - Built an automated corpus builder (`build_corpus.py`) leveraging Wikipedia's public REST API (`/api/rest_v1/page/summary/`) across shared thematic topics (e.g., Bengaluru, Cricket, Water, Mahatma Gandhi, Computer Science) in English (`en`), Hindi (`hi`), Kannada (`kn`), and Tamil (`ta`).
  - Extracted clean, sentence-segmented, NFC-normalized text files saved to `partA/corpus/{eng,hin,kan,tam}.txt`.
  - **Caveat Documented:** Thematic parallel corpus (not 1:1 sentence translated). Evaluates realistic Wikipedia encyclopedic text, though domain differs from conversational code-mixed assistant queries.

---

## Log Entry 3 — Part A2: Script Audit & Bug Isolation Experiments
- **Timestamp:** 2026-09-06 23:14 IST
- **Hypothesis:**
  1. `split(" ")` creates phantom empty-string words on multiple spaces, deflating true fertility.
  2. `.lower()` alters tokenization for acronyms and mixed-case English without affecting Hindi script.
  3. `mean(per_line_ratio)` distorts corpus fertility by giving equal weight to short and long sentences.
  4. `random.seed(1337)` is completely harmless.
- **Experiment:** Built `audit_bugs.py` to isolate each flaw on the starter corpus.
  - Encountered Windows cp1252 terminal encoding error when printing Hindi characters; resolved by enforcing UTF-8 I/O redirection.
- **Measured Results:**
  - **Bug 1 (Whitespace splitting):** 
    - Sample line: `'Please keep the books  in the cupboard.'` contains a double space.
    - `split(" ")` produces 8 words (including `""`); `split()` produces 7 words.
    - Hindi delta: `7.4485` (buggy) vs `7.5985` (fixed) → **+0.1500 fertility distortion**.
    - Direction: Buggy script *underestimates* fertility.
  - **Bug 2 (Lowercasing):**
    - Acronyms like `'NASA and ISRO'` tokenize as 10 tokens in original case vs 11 tokens when lowercased (`'nasa'` splits into subwords).
    - English delta: `1.2831` (lowercased) vs `1.2472` (original) → **-0.0359 distortion**.
    - Hindi delta: `0.0000` (Devanagari has no casing).
    - Direction: Lowercasing artificially inflates English tokens, compressing the reported Hindi/English ratio.
  - **Bug 3 (Mean of ratios vs Corpus aggregate):**
    - English: `1.2831` (mean of ratios) vs `1.2692` (corpus total/total) → **-0.0138 delta**.
    - Hindi: `7.5985` (mean of ratios) vs `7.5246` (corpus total/total) → **-0.0739 delta**.
    - Direction: Mean-of-ratios over-weights short sentences.
  - **Harmless Check (`random.seed`):** 
    - Confirmed `random` is never called during encoding. Delta = `0.0000`.

---

## Log Entry 4 — Part A3: Corrected Multi-Tokenizer & Multi-Denominator Analysis
- **Timestamp:** 2026-09-06 23:23 IST
- **Hypothesis:** Testing on `gpt2` (English-centric) vs `google/muril-base-cased` (Indic-specialized) across 4 denominators (whitespace-word, grapheme cluster, UTF-8 byte, sentence) will prove that high Indic fertility is a tokenizer vocabulary design flaw, not an inherent script property.
- **Experiment 1: Corrected GPT-2 Evaluation (`fertility_fixed.py --tokenizer gpt2`)**
  - **Results:**
    - `eng`: 1.280 tok/word | 0.206 tok/grapheme | 0.206 tok/byte | 26.8 tok/sent
    - `hin`: 8.201 tok/word (6.41×) | 2.420 tok/grapheme | 0.593 tok/byte | 171.4 tok/sent (6.38×)
    - `kan`: 23.255 tok/word (18.17×) | 3.999 tok/grapheme | 0.980 tok/byte | 281.1 tok/sent (10.47×)
    - `tam`: 23.380 tok/word (18.27×) | 4.108 tok/grapheme | 0.994 tok/byte | 305.5 tok/sent (11.38×)
  - **Critical Discovery:** For Kannada and Tamil, `tok/byte ≈ 0.98–0.99`. GPT-2 has zero Dravidian vocabulary merges and represents almost every single raw byte as an independent token!
- **Experiment 2: Indic-Aware Model Evaluation (`fertility_fixed.py --tokenizer hf:google/muril-base-cased`)**
  - **Results:**
    - `eng`: 1.269 tok/word | 0.204 tok/grapheme | 0.204 tok/byte | 26.6 tok/sent
    - `hin`: 1.136 tok/word (**0.89×**) | 0.335 tok/grapheme | 0.082 tok/byte | 23.7 tok/sent (**0.89×**)
    - `kan`: 1.672 tok/word (**1.32×**) | 0.287 tok/grapheme | 0.070 tok/byte | 20.2 tok/sent (**0.76×**)
    - `tam`: 1.514 tok/word (**1.19×**) | 0.266 tok/grapheme | 0.064 tok/byte | 19.8 tok/sent (**0.74×**)
  - **Conclusion:** An Indic-aware vocabulary completely eliminates the 6×–18× penalty. Hindi and Dravidian languages achieve parity with English.
  - **Single Routing Metric:** Tokens per semantic unit (sentence / turn), as words/characters fail cross-lingual equivalence.

---

## Log Entry 5 — Part B: Capacity Reconciliation & KV-Cache Analysis
- **Timestamp:** 2026-09-06 23:19 IST
- **Hypothesis:** The long-context throughput drop at batch 32+ is caused by VRAM KV-cache exhaustion triggering sequence preemption in vLLM.
- **Mathematical Derivation (B1):**
  - Model: 28 layers, 8 KV heads (GQA), head_dim 128, fp16 precision (2 bytes).
  - KV bytes/token = `2 (K+V) × 28 layers × 8 heads × 128 dim × 2 bytes = 114,688 bytes = 112 KB/token`.
  - Usable VRAM on 24GB L4 GPU (`0.92` utilization) = `22.08 GB`.
  - Model weights (4.2B fp16) = `8.40 GB`. Non-KV runtime overhead = `1.60 GB`.
  - Remaining for KV cache = `22.08 - 8.40 - 1.60 = 12.08 GB = 12,970,803,200 bytes`.
  - Max KV tokens = `12.08 GB / 112 KB ≈ 113,097 tokens`.
  - Max concurrent 4096-token sequences = `113,097 / 4,096 ≈ 27.6 → 27–28 sequences`.
- **Log Verification (B1 & B2):**
  - At batch 24 (`24 × 4096 = 98,304 tokens`), `kv_cache_util = 0.93`, `preempted_seqs = 0`. Peak throughput = **1607.4 tok/s**.
  - At batch 32 (`32 × 4096 = 131,072 tokens` > capacity), `kv_cache_util = 0.97`, `preempted_seqs = 7`. Throughput drops to **1384.0 tok/s**!
  - At batch 48, `preempted_seqs = 23`, throughput drops to **1298.5 tok/s**.
- **Report Misreading (B3):**
  - `REPORT_v0` falsely claimed long prompts give better GPU throughput and extrapolated 3200 tok/s at batch 48.
  - `reported_tok_s` includes prefill prompt tokens (87.5% of total).
  - True decode goodput at batch 24 is `(24 × 512) / 61.16 s = 200.9 tok/s` (or via ITL: `24 × (1000 / 96.07 ms) = 249.8 tok/s`).
- **Counter Metric (B4):**
  - Monitor `num_preemptions_total` (vLLM scheduler metric) to catch KV cache thrashing.

---

## Log Entry 6 — Part C: Indic Casual-Tone Adaptation Decision Memo
- **Timestamp:** 2026-09-06 23:22 IST
- **Scenario & Constraints:**
  - 6 Indic languages (Hindi, Kannada, Tamil, Telugu, Bengali, Marathi).
  - Hardware: 1× A100-80GB (2 weeks).
  - Reviewer: 1 native speaker (Hindi + Kannada only, 10 h/week).
  - Launch: 3 weeks. No external API budget.
- **Strategic Recommendation:** Option (a) SFT with LoRA on synthetic pairs, staged with Hindi + Kannada primary launch.
  - Reject (b) 1B rewriter: doubles inference cost/latency per request (catastrophic given Indic token lengths).
  - Reject (c) prompt-only: unreliable style consistency at scale.
- **Arithmetic & Plan:**
  - 12k synthetic pairs (2k/lang) generated on A100 (~24 hours).
  - LoRA fine-tuning on 4B model takes ~18 hours on A100-80GB.
  - 20h reviewer time validates 1,200 Hindi + 1,200 Kannada samples.
  - Explicit kill criterion: If Day-12 human eval on Hindi shows <50% casual rating or COMET drops >3 pts, abort and fall back to prompt engineering.
  - Day-1 experiment: Zero-cost 50-sample prompt test to establish the baseline ceiling.

---
*Notebook verified and sealed for live defense.*
