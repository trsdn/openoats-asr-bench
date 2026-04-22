"""
Build a side-by-side Markdown report from a bench run.

    uv run python compare.py --run-name 12-33-session

Reads `runs/<run-name>/*/{mic,sys}.txt` + `*.metrics.json`, writes
`runs/<run-name>/comparison.md` with per-channel columns, word-count /
RTF / RAM summary, and a simple hallucination heuristic that flags each
model's output for known failure modes (repeated phrases, Whisper's
signature "Thank you for watching" ghosts, etc.).
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


# Known phrases that Whisper regularly hallucinates when audio is silent
# or unclear. English + German variants. Case-insensitive match.
WHISPER_GHOST_PHRASES = [
    r"thank you for watching",
    r"please subscribe",
    r"bitte abonnieren",
    r"untertitel (von|im auftrag)",
    r"untertitelung des zdf",
    r"vielen dank fürs zuschauen",
    r"thanks for watching",
    r"like and subscribe",
]


def count_ghosts(text: str) -> dict[str, int]:
    """Count occurrences of each known Whisper ghost phrase."""
    hits: dict[str, int] = {}
    for pattern in WHISPER_GHOST_PHRASES:
        n = len(re.findall(pattern, text, flags=re.IGNORECASE))
        if n:
            hits[pattern] = n
    return hits


def longest_repeated_ngram(text: str, n: int = 5) -> tuple[str, int]:
    """Return the (tokens, repeat-count) of the most-repeated n-gram with
    length >= n words. Low baseline for normal speech; high counts indicate
    loop / stuttering failures."""
    words = text.split()
    if len(words) < n:
        return "", 0
    ngrams = [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
    counter = Counter(ngrams)
    if not counter:
        return "", 0
    top, count = counter.most_common(1)[0]
    return top, count


def load_metrics(run_dir: Path) -> list[dict]:
    summary = run_dir / "summary.json"
    if summary.exists():
        return json.loads(summary.read_text(encoding="utf-8"))
    # Fallback: glob model dirs
    out: list[dict] = []
    for mdir in sorted(run_dir.iterdir()):
        if not mdir.is_dir():
            continue
        for mfile in mdir.glob("*.metrics.json"):
            out.append(json.loads(mfile.read_text(encoding="utf-8")))
    return out


def build_report(run_dir: Path) -> str:
    records = load_metrics(run_dir)
    if not records:
        return "_No metrics found in this run._\n"

    by_model: dict[str, dict[str, dict]] = {}
    for r in records:
        by_model.setdefault(r["model_id"], {})[r["channel"]] = r

    lines: list[str] = []
    lines.append(f"# ASR bench — {run_dir.name}\n")

    # ──── Performance table ────
    lines.append("## Performance\n")
    lines.append("| Model | Channel | Audio | Wall | RTF | Words | Peak RSS | Error |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---|")
    for model_id in sorted(by_model):
        for ch in sorted(by_model[model_id]):
            r = by_model[model_id][ch]
            lines.append(
                f"| `{model_id}` | {ch} | "
                f"{r['audio_seconds']:.0f}s | "
                f"{r['wall_seconds']:.1f}s | "
                f"{r['rtf']:.2f} | "
                f"{len((r['text'] or '').split())} | "
                f"{r['peak_rss_mb']:.0f} MB | "
                f"{(r.get('error') or '') or '—'} |"
            )
    lines.append("")

    # ──── Hallucination heuristics ────
    lines.append("## Hallucination / degeneration heuristics\n")
    lines.append("| Model | Channel | Whisper ghosts | Top-repeated 5-gram | × |")
    lines.append("|---|---|---|---|---:|")
    for model_id in sorted(by_model):
        for ch in sorted(by_model[model_id]):
            r = by_model[model_id][ch]
            text = r.get("text") or ""
            ghosts = count_ghosts(text)
            gtxt = ", ".join(f"`{p}`×{n}" for p, n in ghosts.items()) or "—"
            ngram, count = longest_repeated_ngram(text)
            ntxt = f"`{ngram}`" if ngram and count > 1 else "—"
            lines.append(f"| `{model_id}` | {ch} | {gtxt} | {ntxt} | {count} |")
    lines.append("")

    # ──── Side-by-side transcripts (first N chars) ────
    preview_chars = 4000
    lines.append(f"## Transcript previews (first ~{preview_chars} chars per cell)\n")

    channels = sorted({ch for m in by_model.values() for ch in m})
    for ch in channels:
        lines.append(f"### Channel: `{ch}`\n")
        header_models = [m for m in sorted(by_model) if ch in by_model[m]]
        lines.append("| " + " | ".join(header_models) + " |")
        lines.append("|" + "|".join(["---"] * len(header_models)) + "|")

        cells: list[str] = []
        for model_id in header_models:
            text = (by_model[model_id][ch].get("text") or "").strip()
            snippet = text[:preview_chars]
            if len(text) > preview_chars:
                snippet += f"… _(+{len(text) - preview_chars} more chars)_"
            # Escape pipes to avoid breaking markdown tables.
            snippet = snippet.replace("|", "\\|").replace("\n", " ")
            cells.append(snippet or "_(empty)_")
        lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    lines.append("---")
    lines.append(
        "_Full transcripts are in each model's subdirectory "
        "(`<model>/<channel>.txt`). Metrics in `summary.json`._"
    )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run-name",
        required=True,
        help="Sub-directory under runs/ produced by bench.py",
    )
    args = ap.parse_args()

    run_dir = Path(__file__).resolve().parent / "runs" / args.run_name
    if not run_dir.is_dir():
        print(f"Run dir not found: {run_dir}")
        return 1

    md = build_report(run_dir)
    out_path = run_dir / "comparison.md"
    out_path.write_text(md, encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
