# flamAI AI Assignment 2026 — The Audit

This repository contains the complete audit of `REPORT_v0.md`, `fertility.py`, and `bench_log.csv` as specified in `AI ASSIGNMENT 2026.pdf`.

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
│   ├── results_gpt2.txt        # GPT-2 benchmark results
│   ├── results_muril.txt       # Indic-aware MuRIL benchmark results
│   ├── routing_memo.md         # A4 ≤1 page Leadership Routing Memo
│   └── PART_A.md               # Part A detailed audit report
├── partB/
│   └── PART_B.md               # Part B capacity reconciliation & KV-cache arithmetic
└── partC/
    └── memo.md                 # Part C casual Indic adaptation decision memo
```

## 🛠️ Quick Verification Commands

```bash
# 1. Run controlled bug isolation experiment (Part A2)
python your-submission/partA/scripts/audit_bugs.py --eng starter_kit/corpus_sample/eng_sample.txt --hin starter_kit/corpus_sample/hin_sample.txt

# 2. Run corrected fertility analysis on multilingual corpus (Part A3)
python your-submission/partA/scripts/fertility_fixed.py --corpus eng=your-submission/partA/corpus/eng.txt --corpus hin=your-submission/partA/corpus/hin.txt --corpus kan=your-submission/partA/corpus/kan.txt --corpus tam=your-submission/partA/corpus/tam.txt --tokenizer gpt2 --compare-buggy
```
