# openoats-asr-bench

Side-by-side transcription benchmark for local open-weight ASR models, run on real OpenOats session audio (mic + system channels).

Currently wired:

| Model | Runtime | Notes |
|---|---|---|
| `parakeet` | NeMo | `nvidia/parakeet-tdt-1.1b` |
| `canary` | NeMo | `nvidia/canary-1b-flash` — multilingual (en/de/fr/es) |
| `whisper-large-v3` | faster-whisper (CTranslate2) | OpenAI Whisper Large-v3, full (not Turbo) |

Voxtral can be added once you have Mistral's HuggingFace access.

## Model + cache location

Copy `.env.example` to `.env` and set the cache paths for your machine before the first run:

```sh
cp .env.example .env
# edit .env — point HF_HOME / NEMO_CACHE_DIR / TORCH_HOME at wherever
# you have 10-15 GB of free space for the weight downloads.
```

`.env` is gitignored (per-machine config). If you skip the copy, `bench.py` falls back to the defaults in `.env.example`, which writes caches into `./model-cache/` next to the repo. First run downloads ~10–15 GB total; subsequent runs reuse the cache.

## Install

```sh
cd /Volumes/big/dev/openoats-asr-bench
uv sync
```

`uv` creates `.venv/` in this directory and resolves all deps. NeMo pulls in a large PyTorch stack; expect 3–5 minutes the first time.

## Run

```sh
# Full matrix against the 12:33 session — this is the default corpus the
# harness was first validated on.
uv run python bench.py \
  --session "$HOME/Library/Application Support/OpenOats/sessions/session_2026-04-22_12-33-37" \
  --run-name 12-33-session

# Just one model / one channel, for a quick smoke test:
uv run python bench.py \
  --session ... \
  --models whisper-large-v3 \
  --channels mic \
  --run-name smoke
```

Each model's output lands in `runs/<run-name>/<model>/{mic,sys}.txt` alongside `*.metrics.json` (wall-clock, RTF, peak RSS, any error caught). `summary.json` at the top of the run directory is the machine-readable matrix.

## Compare

```sh
uv run python compare.py --run-name 12-33-session
```

Writes `runs/<run-name>/comparison.md` with:

- performance table (wall time, realtime factor, peak RAM, error column)
- hallucination heuristics per row (Whisper ghost phrases like *"Thank you for watching"* / *"Bitte abonnieren"*, plus the most-repeated 5-gram and its count — high numbers flag loop degeneration)
- side-by-side transcript previews per channel (first ~4 KB per cell)

Full transcripts are always in the per-model subdirectories; the comparison report is meant to be skim-read.

## Why these three

All three are open-weight, run locally, and represent genuinely different architectures — so we can actually learn something from the comparison instead of benchmarking minor variants of the same approach.

- **Parakeet-TDT** is RNN-T-ish: frame-synchronous, alignment-forced, structurally can't hallucinate on silence. Leads English-only benchmarks at its size.
- **Canary** is attention-encoder-decoder but trained with explicit non-speech / noise tokens; claims low hallucination on silence *and* multilingual coverage (en/de/fr/es). On the Open ASR Leaderboard it frequently beats Whisper on German.
- **Whisper Large-v3** is the reference "big autoregressive decoder" — generous multilingual coverage but the well-known hallucination tendency on silence / music / cross-talk, especially visible on long meeting recordings.

The expected shape of the result: Parakeet + Canary run faster than Whisper on the same hardware, hallucinate less on pauses, and Canary is the one to watch for code-switched DE↔EN meetings.

## Hardware

Tested on Apple Silicon (M2 / M3 / M4). NeMo falls back to CPU for some ops on MPS via `PYTORCH_ENABLE_MPS_FALLBACK=1` (already set in `.env`). faster-whisper runs CPU-only with int8 quantisation — fast enough and sidesteps Metal backend quirks.

Rough wall-clock expectations on M2 Max / 40 min of 16 kHz mono audio per channel:

| Model | Download | First run | Steady state |
|---|---|---|---|
| parakeet | ~4 GB | 5–10 min incl. model load | ~5 min per channel |
| canary | ~4 GB | 5–10 min | ~6 min per channel |
| whisper-large-v3 | ~3 GB | 10–15 min | ~10 min per channel |

Running the full matrix (3 models × 2 channels) on a 40-min session will take roughly one hour end-to-end the first time. Plan accordingly.
