#!/usr/bin/env python3
"""
fertility_fixed.py — A2/A3: Corrected tokenizer fertility analysis.

Fixes vs original fertility.py:
  Fix 1: word splitting  — use str.split() not str.split(" ")
  Fix 2: do NOT lowercase — evaluate on original text to match real serving
  Fix 3: corpus-level aggregation — total_tokens / total_words (not mean of ratios)

New features for A3:
  - Supports multiple tokenizers: gpt2 (tiktoken) + any HF tokenizer
  - Multiple denominators: per whitespace-word, per grapheme cluster, per UTF-8 byte
  - Per-sentence denominator (most meaningful for routing cost)

Usage:
  # Single tokenizer, default denominators
  python fertility_fixed.py \\
      --corpus eng=corpus/eng.txt \\
      --corpus hin=corpus/hin.txt \\
      --corpus kan=corpus/kan.txt \\
      --corpus tam=corpus/tam.txt \\
      --tokenizer gpt2

  # With a multilingual HF tokenizer
  python fertility_fixed.py \\
      --corpus eng=corpus/eng.txt \\
      --corpus hin=corpus/hin.txt \\
      --corpus kan=corpus/kan.txt \\
      --corpus tam=corpus/tam.txt \\
      --tokenizer hf:ai4bharat/indic-bert

Author: fixed version for FlamAI audit
"""

import argparse
import unicodedata
import sys

# ---------------------------------------------------------------------------
# Grapheme cluster helper (uses 'regex' if available, else Unicode scalar fallback)
# ---------------------------------------------------------------------------
try:
    import regex as re_mod
    def grapheme_clusters(text: str) -> list[str]:
        return re_mod.findall(r'\X', text)
except ImportError:
    # Fallback: count Unicode codepoints (slightly different from grapheme clusters
    # for combined characters, but close enough without the regex package)
    def grapheme_clusters(text: str) -> list[str]:  # type: ignore[misc]
        return list(text)


# ---------------------------------------------------------------------------
# Tokenizer loaders
# ---------------------------------------------------------------------------
def load_tokenizer(spec: str):
    """Return an encode function: str -> list[int]."""
    if spec.startswith("hf:"):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(spec[3:])
        return lambda s: tok.encode(s, add_special_tokens=False)
    else:
        import tiktoken
        enc = tiktoken.get_encoding(spec)
        return enc.encode


# ---------------------------------------------------------------------------
# Corpus reader
# ---------------------------------------------------------------------------
def read_lines(path: str) -> list[str]:
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            line = unicodedata.normalize("NFC", line)
            lines.append(line)
    return lines


# ---------------------------------------------------------------------------
# Core analysis — FIX 1, FIX 2, FIX 3 applied
# ---------------------------------------------------------------------------
def analyze(lines: list[str], encode) -> dict:
    """
    Returns a dict with corpus-level metrics (not per-line averages):
      - fertility_word  : tokens / whitespace-word  (corpus-level)
      - fertility_graph : tokens / grapheme-cluster (corpus-level)
      - fertility_byte  : tokens / UTF-8 byte       (corpus-level)
      - fertility_sent  : tokens / sentence         (corpus-level, = avg tokens/sentence)
    """
    total_tokens = 0
    total_words  = 0
    total_graphs = 0
    total_bytes  = 0
    total_sents  = len(lines)

    for line in lines:
        # FIX 2: NO lowercasing — evaluate on original text as-served
        tokens = encode(line)
        n_tok  = len(tokens)

        # FIX 1: split() not split(" ") — handles multiple/leading/trailing spaces
        words  = line.split()
        n_word = len(words) if words else 1  # guard for empty lines (shouldn't occur)

        n_graph = len(grapheme_clusters(line))
        n_byte  = len(line.encode("utf-8"))

        total_tokens += n_tok
        total_words  += n_word
        total_graphs += n_graph
        total_bytes  += n_byte

    # FIX 3: corpus-level ratios (not mean of per-line ratios)
    return {
        "fertility_word"  : total_tokens / total_words,
        "fertility_graph" : total_tokens / total_graphs,
        "fertility_byte"  : total_tokens / total_bytes,
        "fertility_sent"  : total_tokens / total_sents,
        "total_tokens"    : total_tokens,
        "total_words"     : total_words,
        "total_sents"     : total_sents,
    }


# ---------------------------------------------------------------------------
# Original buggy analysis (for comparison/evidence)
# ---------------------------------------------------------------------------
def analyze_buggy(lines: list[str], encode) -> dict:
    """Replicates the ORIGINAL fertility.py logic (bugs included) for comparison."""
    per_line_fertility = []
    per_line_tpc = []
    for line in lines:
        line_lower = line.lower()                          # BUG 2: lowercasing
        tokens = encode(line_lower)
        words  = line_lower.split(" ")                     # BUG 1: split(" ")
        chars  = len(line_lower)
        per_line_fertility.append(len(tokens) / len(words))
        per_line_tpc.append(len(tokens) / chars)
    n = len(per_line_fertility)
    return {
        "fertility_word" : sum(per_line_fertility) / n,   # BUG 3: mean of ratios
        "tok_per_char"   : sum(per_line_tpc) / n,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", action="append", required=True, metavar="LANG=PATH")
    ap.add_argument("--tokenizer", default="gpt2", help="gpt2 | hf:<repo_id>")
    ap.add_argument(
        "--compare-buggy", action="store_true",
        help="Also run the original buggy analysis and show delta"
    )
    args = ap.parse_args()

    encode = load_tokenizer(args.tokenizer)
    print(f"\ntokenizer : {args.tokenizer}")

    # Header
    hdr = f"{'lang':<8}{'tok/word':>12}{'tok/graph':>12}{'tok/byte':>12}{'tok/sent':>12}"
    print(hdr)
    print("-" * len(hdr))

    results = {}
    for spec in args.corpus:
        lang, path = spec.split("=", 1)
        lines = read_lines(path)
        res   = analyze(lines, encode)
        results[lang] = res
        print(
            f"{lang:<8}"
            f"{res['fertility_word']:>12.3f}"
            f"{res['fertility_graph']:>12.3f}"
            f"{res['fertility_byte']:>12.3f}"
            f"{res['fertility_sent']:>12.1f}"
        )

    if len(results) >= 2:
        langs = list(results)
        base  = langs[0]
        print()
        print("Relative to", base, "(cross-language cost ratios):")
        for lang in langs[1:]:
            r_word  = results[lang]["fertility_word"]  / results[base]["fertility_word"]
            r_sent  = results[lang]["fertility_sent"]  / results[base]["fertility_sent"]
            print(f"  {lang}: {r_word:.2f}x per-word  |  {r_sent:.2f}x per-sentence")

    if args.compare_buggy:
        print("\n--- BUGGY (original fertility.py) for comparison ---")
        hdr2 = f"{'lang':<8}{'tok/word (buggy)':>20}{'tok/char (buggy)':>20}"
        print(hdr2)
        print("-" * len(hdr2))
        for spec in args.corpus:
            lang, path = spec.split("=", 1)
            lines = read_lines(path)
            buggy = analyze_buggy(lines, encode)
            fixed = results[lang]
            delta = fixed["fertility_word"] - buggy["fertility_word"]
            print(
                f"{lang:<8}"
                f"{buggy['fertility_word']:>20.3f}"
                f"{buggy['tok_per_char']:>20.3f}"
                f"   (fixed delta: {delta:+.3f})"
            )


if __name__ == "__main__":
    main()
