#!/usr/bin/env python3
"""
make_corpus.py — Build a FLORES-200 style parallel corpus from Wikipedia sentences.

We use Wikipedia's REST API (no authentication) to fetch clean article text 
in multiple languages and extract 250 representative sentences per language.

Languages: English, Hindi, Kannada, Tamil
This gives us a real multilingual corpus without HuggingFace auth requirements.

Usage:
    python make_corpus.py --output ../corpus/
"""

import argparse
import json
import re
import time
import unicodedata
import urllib.request
import urllib.parse
from pathlib import Path

# Wikipedia articles to pull text from — chosen to have translations across all 4 languages
ARTICLES = {
    "eng": [
        "Mahatma_Gandhi", "Bengaluru", "Indian_cuisine", "Cricket", 
        "Solar_System", "Water", "Computer_science", "Human_rights",
        "Climate_change", "Mathematics", "India", "Yoga",
    ],
    "hin": [
        "महात्मा_गांधी", "बेंगलुरु", "भारतीय_व्यंजन", "क्रिकेट",
        "सौर_मण्डल", "जल", "कम्प्यूटर_विज्ञान", "मानव_अधिकार",
        "जलवायु_परिवर्तन", "गणित", "भारत", "योग",
    ],
    "kan": [
        "ಮಹಾತ್ಮ_ಗಾಂಧಿ", "ಬೆಂಗಳೂರು", "ಭಾರತೀಯ_ಪಾಕಪದ್ಧತಿ", "ಕ್ರಿಕೆಟ್",
        "ಸೌರವ್ಯೂಹ", "ನೀರು", "ಕಂಪ್ಯೂಟರ್_ವಿಜ್ಞಾನ", "ಮಾನವ_ಹಕ್ಕುಗಳು",
        "ಹವಾಮಾನ_ಬದಲಾವಣೆ", "ಗಣಿತ", "ಭಾರತ", "ಯೋಗ",
    ],
    "tam": [
        "மகாத்மா_காந்தி", "பெங்களூரு", "இந்திய_உணவு", "கிரிக்கெட்",
        "சூரிய_குடும்பம்", "நீர்", "கணினி_அறிவியல்", "மனித_உரிமைகள்",
        "காலநிலை_மாற்றம்", "கணிதம்", "இந்தியா", "யோகா",
    ],
}

WIKI_LANG = {
    "eng": "en",
    "hin": "hi",
    "kan": "kn",
    "tam": "ta",
}


def fetch_wikipedia_text(lang_code: str, title: str, max_retries: int = 2) -> str:
    """Fetch plain text of a Wikipedia article via the REST API."""
    wiki_lang = WIKI_LANG[lang_code]
    encoded = urllib.parse.quote(title, safe="")
    url = f"https://{wiki_lang}.wikipedia.org/api/rest_v1/page/summary/{encoded}"
    headers = {"User-Agent": "FlamAI-corpus-builder/1.0 (educational-assignment)"}
    for attempt in range(max_retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("extract", "")
        except Exception as e:
            if attempt < max_retries:
                time.sleep(1.5)
            else:
                print(f"    [WARN] Failed {lang_code}/{title}: {e}")
                return ""


def extract_sentences(text: str, lang: str) -> list[str]:
    """Extract clean sentences from Wikipedia text."""
    if not text:
        return []
    text = unicodedata.normalize("NFC", text)
    # Split on sentence boundaries
    if lang in ("eng",):
        # Western: split on period/!/?
        sents = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    else:
        # For Indic scripts, split on danda (।), period, or newline
        sents = re.split(r'[।\n.!?]+', text)
    
    cleaned = []
    for s in sents:
        s = s.strip()
        # Filter: min 10 chars, max 500 chars, no URLs, no [citations]
        if len(s) < 10 or len(s) > 500:
            continue
        if "http" in s or "[" in s or "{" in s:
            continue
        # Require at least some alphabetic content
        alpha = sum(1 for c in s if c.isalpha())
        if alpha < 8:
            continue
        cleaned.append(s)
    return cleaned


def build_corpus(output_dir: Path, target_per_lang: int = 250):
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = {}

    for lang, articles in ARTICLES.items():
        print(f"\n[*] Collecting {lang} sentences ...")
        all_sentences = []
        for title in articles:
            time.sleep(0.3)  # be polite to Wikipedia
            text = fetch_wikipedia_text(lang, title)
            sents = extract_sentences(text, lang)
            print(f"    {title}: {len(sents)} sentences")
            all_sentences.extend(sents)

        # Deduplicate
        seen = set()
        unique = []
        for s in all_sentences:
            key = s.strip().lower()
            if key not in seen:
                seen.add(key)
                unique.append(s)

        # Take up to target_per_lang
        final = unique[:target_per_lang]
        out_path = output_dir / f"{lang}.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            for s in final:
                f.write(s.strip() + "\n")

        meta[lang] = {
            "source": "Wikipedia REST API (article summaries)",
            "articles": articles,
            "raw_sentences": len(all_sentences),
            "unique_sentences": len(unique),
            "final_sentences": len(final),
            "path": str(out_path),
        }
        print(f"  → {len(final)} unique sentences written to {out_path}")

    meta_path = output_dir / "corpus_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\n[✓] Corpus metadata saved to {meta_path}")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="corpus")
    ap.add_argument("--target", type=int, default=250, help="Target sentences per language")
    args = ap.parse_args()
    build_corpus(Path(args.output), args.target)
    print("\n--- Corpus Caveats ---")
    print("Source       : Wikipedia article summaries (REST API, CC-BY-SA)")
    print("Languages    : English (eng), Hindi (hin), Kannada (kan), Tamil (tam)")
    print("Domain       : Encyclopedic prose — not conversational")
    print("Parallel     : THEMATIC (same articles, not sentence-aligned)")
    print("              NOTE: unlike FLORES-200, sentences are NOT 1:1 translations.")
    print("              Per-sentence ratios must use tok/sent not cross-language alignment.")
    print("Caveats      :")
    print("  1. Not parallel: tok/sentence ratios are intra-language, not cross-language aligned.")
    print("  2. Domain mismatch vs real assistant queries (formal encyclopedia vs casual chat).")
    print("  3. Wikipedia summaries: 1-3 paragraphs; may underrepresent long complex sentences.")
    print("  4. Sentence splitting heuristic: may produce fragments near headers or lists.")


if __name__ == "__main__":
    main()
