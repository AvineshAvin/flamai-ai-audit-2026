#!/usr/bin/env python3
"""
audit_bugs.py — A2: Isolate each bug in fertility.py and measure its effect.

This script runs controlled experiments to prove each bug claim with numbers.
Run this FIRST to generate the evidence table for the A2 write-up.

Usage:
    python audit_bugs.py --eng ../../starter_kit/corpus_sample/eng_sample.txt \\
                         --hin ../../starter_kit/corpus_sample/hin_sample.txt
"""

import argparse
import unicodedata
import sys


def load_gpt2():
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    return enc.encode


def read_lines(path):
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            line = unicodedata.normalize("NFC", line)
            lines.append(line)
    return lines


# ============================================================
# Bug 1: split(" ") vs split()
# ============================================================
def experiment_bug1(lines, encode, lang):
    """Show effect of single-space vs any-whitespace word splitting."""
    results_buggy = []
    results_fixed = []
    double_space_lines = []

    for line in lines:
        line_l = line.lower()
        tokens = encode(line_l)
        n_tok = len(tokens)

        words_buggy = line_l.split(" ")     # BUG: single space
        words_fixed = line_l.split()        # FIX: any whitespace

        results_buggy.append(n_tok / len(words_buggy))
        results_fixed.append(n_tok / len(words_fixed))

        if len(words_buggy) != len(words_fixed):
            double_space_lines.append((line, len(words_buggy), len(words_fixed)))

    buggy_mean = sum(results_buggy) / len(results_buggy)
    fixed_mean = sum(results_fixed) / len(results_fixed)

    print(f"\n=== BUG 1: split(' ') vs split() — lang={lang} ===")
    print(f"  Lines with double-spaces (affected): {len(double_space_lines)}")
    for ln, wb, wf in double_space_lines:
        print(f"    >{ln!r}  buggy_words={wb}  fixed_words={wf}")
    print(f"  fertility (buggy split) : {buggy_mean:.4f}")
    print(f"  fertility (fixed split) : {fixed_mean:.4f}")
    print(f"  DELTA (fixed - buggy)  : {fixed_mean - buggy_mean:+.4f}")
    print(f"  Direction: buggy UNDER-estimates fertility (empty string phantom words inflate denominator)")


# ============================================================
# Bug 2: lowercasing distortion
# ============================================================
def experiment_bug2(lines, encode, lang):
    """Show effect of lowercasing on token count."""
    tok_lower  = [len(encode(l.lower())) for l in lines]
    tok_orig   = [len(encode(l))         for l in lines]

    words = [len(l.split()) for l in lines]
    words_nz = [max(w, 1) for w in words]

    fert_lower = sum(t / w for t, w in zip(tok_lower, words_nz)) / len(lines)
    fert_orig  = sum(t / w for t, w in zip(tok_orig,  words_nz)) / len(lines)

    changed = [(lines[i], tok_lower[i], tok_orig[i])
               for i in range(len(lines)) if tok_lower[i] != tok_orig[i]]

    print(f"\n=== BUG 2: .lower() distortion — lang={lang} ===")
    print(f"  Lines where token count changes after .lower(): {len(changed)}")
    for ln, tl, to in changed[:5]:
        print(f"    >{ln!r}  tokens_lower={tl}  tokens_orig={to}  delta={tl-to:+d}")
    print(f"  fertility (with .lower()) : {fert_lower:.4f}")
    print(f"  fertility (original text) : {fert_orig:.4f}")
    print(f"  DELTA (orig - lower)     : {fert_orig - fert_lower:+.4f}")
    print(f"  Direction: lowercasing UNDER-estimates English fertility for mixed-case/acronym text")


# ============================================================
# Bug 3: mean of per-line ratios vs corpus-level ratio
# ============================================================
def experiment_bug3(lines, encode, lang):
    """Show effect of averaging per-line ratios vs corpus-level total/total."""
    per_line_fert = []
    total_tok  = 0
    total_word = 0

    for line in lines:
        line_l = line.lower()
        tokens = encode(line_l)
        words  = line_l.split()
        if not words:
            continue
        per_line_fert.append(len(tokens) / len(words))
        total_tok  += len(tokens)
        total_word += len(words)

    buggy_mean = sum(per_line_fert) / len(per_line_fert)
    fixed_ratio = total_tok / total_word

    print(f"\n=== BUG 3: mean(per-line fertility) vs total_tokens/total_words — lang={lang} ===")
    print(f"  fertility (mean of ratios) : {buggy_mean:.4f}  ← REPORT value")
    print(f"  fertility (corpus-level)   : {fixed_ratio:.4f}  ← CORRECT")
    print(f"  DELTA (correct - report)  : {fixed_ratio - buggy_mean:+.4f}")
    print(f"  Direction: mean-of-ratios gives MORE weight to short sentences, biasing result")


# ============================================================
# Harmless: random.seed()
# ============================================================
def note_harmless():
    print(f"\n=== HARMLESS: random.seed(1337) ===")
    print(f"  random.seed(1337) is set at module level but 'random' is never called.")
    print(f"  Token counts and fertilities are fully deterministic — no sampling is done.")
    print(f"  Effect on reported numbers: ZERO.")
    print(f"  This looks suspicious (why seed randomness if not used?) but is safe.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eng", required=True, help="Path to English corpus file")
    ap.add_argument("--hin", required=True, help="Path to Hindi corpus file")
    args = ap.parse_args()

    encode = load_gpt2()
    eng_lines = read_lines(args.eng)
    hin_lines = read_lines(args.hin)

    print("=" * 60)
    print("AUDIT EXPERIMENTS — fertility.py bug isolation")
    print("Tokenizer: gpt2 (tiktoken)")
    print("=" * 60)

    experiment_bug1(eng_lines, encode, "eng")
    experiment_bug1(hin_lines, encode, "hin")

    experiment_bug2(eng_lines, encode, "eng")
    experiment_bug2(hin_lines, encode, "hin")

    experiment_bug3(eng_lines, encode, "eng")
    experiment_bug3(hin_lines, encode, "hin")

    note_harmless()

    print("\n" + "=" * 60)
    print("Summary of distortion directions:")
    print("  Bug 1 (split): fertility is UNDER-estimated (phantom empty words)")
    print("  Bug 2 (lower): English fertility slightly UNDER-estimated")
    print("  Bug 3 (mean) : direction depends on sentence length distribution")
    print("All three bugs compound — the reported Hindi/English ratio is unreliable.")
    print("=" * 60)


if __name__ == "__main__":
    main()
