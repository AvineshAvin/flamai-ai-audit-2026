# Part C — Decision Memo: Making Indic Responses Sound Casual

**Date:** 2026-09-06  
**From:** Candidate  
**To:** FlamAI Product / Engineering Leadership  
**Re:** Recommendation on casual-tone adaptation for Hindi, Kannada, Tamil, Telugu, Bengali, Marathi  

---

## Problem Statement

Our assistant's Indic-language outputs are described as "too formal/textbook." We need responses to sound casual and conversational in 6 languages: Hindi, Kannada, Tamil, Telugu, Bengali, Marathi.

**Resources available:**
- 1× A100-80GB GPU, 2 weeks
- 1 native-speaker reviewer (Hindi + Kannada), 10 h/week
- Launch review in 3 weeks; no external API budget

---

## Assumptions

1. "Casual" is a style adaptation, not a domain shift. The model already understands these languages; we need it to pick the right register.
2. The A100-80GB can fine-tune a ~4B model (FLM-4B-Instruct) in bf16 with LoRA, or can serve a separate 1B rewriter.
3. Synthetic data generation can be done on the same A100 (using the main model as a synthetic-data generator in off-hours).
4. The 3-week launch review means code+eval must be complete in 18 days (3 days buffer for prep).
5. "Casual" can be operationalized as a shift in formality score measurable by an automatic metric (e.g., LENS, or a trained classifier) plus human review on a held-out set.

---

## My Recommendation: Option (a) — SFT with LoRA on synthetic pairs

**Primary choice: SFT pass on synthetic "casualized" response pairs, scoped to Hindi only for launch.**

### Reasoning

| Option | Feasibility | Quality | Risk |
|--------|------------|---------|------|
| **(a) SFT w/ LoRA** | High — fits A100 easily | High — register learned in weights | Medium — data quality risk |
| (b) 1B rewriter | High — low latency | Medium — extra latency per request | High — serving cost doubles for Indic |
| (c) Prompt-only | Very high — zero cost | Low — inconsistent at scale | High — model may not comply reliably |

Option (b) doubles inference cost for every Indic request (main model + rewriter in series) and adds latency. Part A showed Hindi costs 6× more tokens than English — adding another inference step is expensive. Option (c) has been tried by other teams and reliably fails for style at scale; it cannot be the primary approach.

SFT with LoRA:
- A100-80GB fits FLM-4B-Instruct LoRA training comfortably (bf16 base + 4-bit quantized base + fp16 adapters ≈ ~18 GB with batch size 4)
- LoRA adds only inference-time adapter loading (negligible cost)
- Quality can be validated by reviewer before launch

### Back-of-Envelope Arithmetic

**Data volume:**
- Generate 2,000 (formal → casual) Hindi pairs × 6 languages = 12,000 pairs total
- Generation rate: ~500 pairs/hour on A100 using the main model in batch mode
- Time to generate: 12,000 / 500 = **24 GPU-hours** ≈ 1 day

**Training cost:**
- LoRA fine-tune on 12,000 examples × 3 epochs, seq_len 512:
- ~12,000 × 3 × 512 / 1M tokens/s (A100 fp16 throughput) ≈ **18 GPU-hours** ≈ 18 hours
- Total: data gen + training ≈ 2 days

**Reviewer throughput:**
- 10 h/week × 2 weeks = 20 hours of reviewer time
- Hindi + Kannada coverage (reviewer only covers these 2/6 languages)
- At ~60 samples/hour: reviewer can validate 1,200 samples for Hindi and 1,200 for Kannada
- **Gap:** Telugu, Tamil, Bengali, Marathi have NO native reviewer → data quality risk

**Launch scope decision:** Ship Hindi + Kannada at launch (reviewer-validated). Flag Telugu/Tamil/Bengali/Marathi as "improved but unvalidated" until reviewer coverage is added.

---

## Success Metric

**Primary metric:** Formality score on a 100-sentence held-out casual-query benchmark.

> **Threshold:** ≥ 70% of responses rated "appropriately casual" (≤ 2 on a 5-point formality scale) by the Hindi/Kannada reviewer on the held-out set, compared to ≤ 20% before the SFT.

**Secondary metric:** BLEU/COMET against human casual references (automatic, no reviewer time) — should not regress below baseline.

---

## Kill Criterion

> **Abandon this path if, by Day 12 (end of Week 2):**
> - Human eval on the first 200 Hindi samples shows < 50% of responses rated casual (insufficient improvement); OR
> - Automatic formality classifier scores show < 30% shift from baseline; OR
> - COMET on a general Hindi benchmark drops > 3 points vs the base model (catastrophic forgetting)

If any criterion fires, fall back to **Option (c) — prompt engineering** as a stopgap for launch while we investigate the data quality failure.

---

## Day-1 Experiment

**Before training anything, on Day 1:**

1. Write a zero-shot prompt: *"Reply to the following in casual, everyday Hindi (use 'tum' not 'aap', use short sentences, use common contractions):"* and run it on 50 representative queries.

2. Have the reviewer score those 50 outputs (takes ~1 hour).

**Goal:** Measure the prompt-engineering ceiling. If prompt-only already achieves 60%+ casual rating, reconsider the SFT path — we might get the launch win without training. If it achieves < 40%, proceed with SFT. The day-1 experiment costs zero GPU-hours and de-risks the entire 2-week plan.

---

## Timeline

| Day | Action |
|-----|--------|
| 1   | Day-1 prompt experiment (50 samples, reviewer scores) |
| 2–3 | Design data generation prompts; generate 2K Hindi casual pairs |
| 4   | Reviewer validates 200 Hindi samples; adjust generation prompt if needed |
| 5–7 | Generate full 12K dataset (all 6 languages) |
| 8–9 | LoRA fine-tune (18 hours on A100) |
| 10–11 | Eval: held-out set, formality classifier, COMET check |
| 12  | Kill/go decision gate |
| 13–14 | Fix top failure modes; re-eval |
| 15  | Final reviewer sign-off on Hindi + Kannada |
| 16–18 | Integration, launch prep, buffer |
