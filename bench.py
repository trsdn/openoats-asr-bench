"""
openoats-asr-bench — side-by-side ASR model comparison on OpenOats session audio.

Usage:

    uv run python bench.py \
        --session ~/Library/Application\\ Support/OpenOats/sessions/session_2026-04-22_12-33-37 \
        --models parakeet canary whisper-large-v3 \
        --run-name 12-33-session

Each model runs once on both `mic.caf` and `sys.caf` from the session
folder; transcript + metrics (wall-clock, RTF, peak RAM) land under
`runs/<run-name>/<model>/`.

Design notes
------------
- All HF / NeMo / torch caches are redirected to `/Volumes/big/aimodels/`
  via the `.env` file next to this script so `~/` doesn't fill up with
  model weights.
- Each model runs in its own function, isolated so a failure in one
  doesn't kill the rest of the matrix.
- We avoid a shared abstraction because the three runtimes (NeMo,
  faster-whisper) have different load / infer signatures and smashing
  them into one interface obscures more than it saves.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

# Load .env (or fall back to .env.example) before anything that might
# touch HF / NeMo / Torch caches. `.env` is per-machine and gitignored;
# `.env.example` is the committed template.
_repo_dir = Path(__file__).resolve().parent
_env_file = _repo_dir / ".env"
if not _env_file.exists():
    _example = _repo_dir / ".env.example"
    if _example.exists():
        print(
            f"[openoats-asr-bench] no .env found — falling back to {_example.name}. "
            f"Copy it to .env and edit paths for your machine.",
            file=sys.stderr,
        )
        _env_file = _example
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

console = Console()


# ──────────────────────────────────────────────
# Audio loading
# ──────────────────────────────────────────────

TARGET_SR = 16_000


def load_mono_16k(path: Path) -> np.ndarray:
    """Read `path`, downmix to mono, resample to 16 kHz. Returns float32
    in [-1, 1].

    First tries libsndfile (via soundfile); some CAF variants written by
    AVAudioFile aren't parseable there, so we fall back to ffmpeg which
    handles every CAF codec in the wild. ffmpeg is a hard runtime
    dependency — install via `brew install ffmpeg` on macOS."""
    try:
        audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
        if audio.shape[1] > 1:
            audio = audio.mean(axis=1)
        else:
            audio = audio[:, 0]
    except Exception as libsnd_err:
        audio, sr = _ffmpeg_decode(path)
        if audio is None:
            raise RuntimeError(
                f"Could not decode {path} via soundfile or ffmpeg: {libsnd_err}"
            )
    if sr != TARGET_SR:
        import librosa  # lazy — librosa pulls in scipy etc.
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    return audio.astype(np.float32, copy=False)


def _ffmpeg_decode(path: Path) -> tuple[np.ndarray | None, int]:
    """Decode `path` to mono float32 16 kHz via a single ffmpeg subprocess
    call. Returns (samples, sample_rate) or (None, 0) on failure."""
    import shutil
    import subprocess

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return None, 0

    cmd = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(path),
        "-ac", "1",             # downmix to mono
        "-ar", str(TARGET_SR),  # resample to 16 kHz
        "-f", "f32le",          # raw float32 little-endian
        "pipe:1",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"[ffmpeg] failed: {exc.stderr.decode(errors='replace')}", file=sys.stderr)
        return None, 0
    samples = np.frombuffer(proc.stdout, dtype=np.float32)
    return samples, TARGET_SR


# ──────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────


@dataclass
class ModelRun:
    model_id: str
    channel: str        # "mic" | "sys"
    audio_seconds: float
    wall_seconds: float
    rtf: float           # wall / audio (< 1 = faster than realtime)
    peak_rss_mb: float
    text: str
    error: str | None = None

    def to_json(self) -> dict:
        return asdict(self)


def peak_rss_mb() -> float:
    """Peak resident set size since process start, in MB (Darwin reports
    ru_maxrss in bytes; Linux reports kB)."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / (1024 * 1024)
    return rss / 1024


# ──────────────────────────────────────────────
# Model runners
# ──────────────────────────────────────────────


def run_faster_whisper(audio: np.ndarray, model_name: str = "large-v3") -> str:
    from faster_whisper import WhisperModel
    # int8 for Apple Silicon — fast enough, small footprint. float16 is
    # not supported on CPU; the Metal backend is experimental in CT2.
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    segments, _info = model.transcribe(
        audio,
        beam_size=5,
        vad_filter=True,  # trims long silence → fewer hallucinations
        language=None,    # auto-detect
    )
    chunks = [seg.text.strip() for seg in segments]
    del model
    gc.collect()
    return " ".join(chunks).strip()


def run_nemo_asr(
    audio: np.ndarray,
    model_id: str,
    extra_transcribe_kwargs: dict | None = None,
) -> str:
    """Unified runner for Parakeet / Canary. Chunks the audio into short
    windows and transcribes each one individually — NeMo's default
    Lhotse dataloader fails on long audio with macOS's spawn-based
    multiprocessing (the 8 default workers silently exit before producing
    a single batch, leaving "Transcribing: 0it" in the log). Passing each
    window as its own one-shot transcribe call sidesteps the dataloader
    path entirely and keeps peak RAM bounded.

    `extra_transcribe_kwargs` is merged into `transcribe()` — used for
    Canary's `source_lang`/`target_lang`/`pnc`/`task` hints."""
    from nemo.collections.asr.models import ASRModel
    import tempfile

    extra_kwargs = dict(extra_transcribe_kwargs or {})
    model = ASRModel.from_pretrained(model_id)
    # Belt-and-braces: even if NeMo's transcribe() internally builds a
    # dataloader, make sure any worker count is 0.
    try:
        model._cfg.test_ds.num_workers = 0
    except Exception:
        pass
    try:
        model._cfg.validation_ds.num_workers = 0
    except Exception:
        pass

    chunk_seconds = 30.0
    chunk_samples = int(chunk_seconds * TARGET_SR)
    n_chunks = max(1, (len(audio) + chunk_samples - 1) // chunk_samples)

    texts: list[str] = []
    tmpdir = Path(tempfile.mkdtemp(prefix="nemo-bench-"))
    try:
        for i in range(n_chunks):
            start = i * chunk_samples
            end = min(len(audio), start + chunk_samples)
            segment = audio[start:end]
            if len(segment) < TARGET_SR // 10:
                # Skip anything shorter than 100 ms — NeMo occasionally
                # errors on micro-chunks at the end of the file.
                continue
            wav_path = tmpdir / f"chunk_{i:05d}.wav"
            sf.write(str(wav_path), segment, TARGET_SR, subtype="PCM_16")
            try:
                result = model.transcribe(
                    [str(wav_path)],
                    batch_size=1,
                    num_workers=0,
                    verbose=False,
                    **extra_kwargs,
                )
            except TypeError:
                # Older NeMo versions don't accept num_workers/verbose kwargs.
                # Still try with the model-specific extras (Canary's
                # source_lang/target_lang are required, not optional).
                result = model.transcribe(
                    [str(wav_path)],
                    batch_size=1,
                    **extra_kwargs,
                )
            finally:
                wav_path.unlink(missing_ok=True)

            # NeMo returns list[str] or list[Hypothesis] depending on
            # the model family + version. Normalise.
            for r in result:
                if isinstance(r, str):
                    texts.append(r)
                elif hasattr(r, "text"):
                    texts.append(str(r.text))
                elif isinstance(r, list) and r and hasattr(r[0], "text"):
                    texts.append(str(r[0].text))
                else:
                    texts.append(str(r))
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
        del model
        gc.collect()

    return " ".join(t.strip() for t in texts).strip()


MODEL_REGISTRY: dict[str, dict] = {
    "parakeet-live": {
        # Reads OpenOats's own transcript.live.jsonl — the Parakeet v3
        # output the user actually sees in production. Avoids having to
        # reproduce the FluidAudio / Swift pipeline in Python (NeMo's
        # EncDecRNNTBPE wrapper for parakeet-tdt-0.6b-v3 multilingual
        # decoded to 0 tokens on our test audio). This is also a
        # "truer" comparison target: it's literally the output the
        # user is comparing Whisper / Canary against.
        "kind": "openoats_live",
        "label": "OpenOats live Parakeet-TDT v3 (as produced by the app)",
    },
    "canary": {
        "kind": "nemo",
        "nemo_id": "nvidia/canary-1b-flash",
        "label": "NVIDIA Canary-1B-Flash (multilingual: en/de/fr/es)",
        # Canary is a Multi-Task model: without explicit language hints
        # it auto-translates to English. For an honest ASR comparison on
        # DE meetings we pin both source+target to German and request
        # punctuation+capitalisation. Mixed-language utterances are a
        # known weakness of this config — live Parakeet handles code-
        # switching better, but this is the cleanest Canary-on-German
        # baseline for the benchmark.
        "nemo_transcribe_kwargs": {
            "source_lang": "de",
            "target_lang": "de",
            "pnc": "yes",
            "task": "asr",
        },
    },
    "whisper-large-v3": {
        "kind": "whisper",
        "fw_id": "large-v3",
        "label": "OpenAI Whisper Large-v3 (via faster-whisper)",
    },
}


def run_openoats_live(session_dir: Path, channel: str) -> str:
    """Extract the Parakeet-v3 output from OpenOats's own
    `transcript.live.jsonl` and split it by speaker so we can line it up
    against model output for the `mic` vs `sys` channels separately.

    `.you` speaker records map to the mic channel, everything else
    (`.them` / `.remote(…)`) maps to sys. If the preferred file name is
    missing we fall back to the `.pre-cleanup.bak` copy the app writes
    at finalize time."""
    import json

    candidates = [
        session_dir / "transcript.live.jsonl",
        session_dir / "transcript.live.jsonl.pre-cleanup.bak",
    ]
    jsonl_path = next((p for p in candidates if p.exists()), None)
    if jsonl_path is None:
        raise FileNotFoundError(
            f"No transcript.live.jsonl (or .bak) under {session_dir}"
        )

    mic_parts: list[str] = []
    sys_parts: list[str] = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        speaker = rec.get("speaker")
        # Speaker is serialised as either "you" or a dict like
        # {"them": null} / {"remote": 1} depending on OpenOats version.
        if speaker == "you" or (isinstance(speaker, dict) and "you" in speaker):
            is_mic = True
        else:
            is_mic = False
        # Prefer the cleaned / refined text (`refinedText`) over the raw
        # `text` — that's what the UI shows.
        text = rec.get("refinedText") or rec.get("text") or ""
        text = text.strip()
        if not text:
            continue
        (mic_parts if is_mic else sys_parts).append(text)

    return " ".join(mic_parts if channel == "mic" else sys_parts).strip()


def run_model(
    model_key: str,
    audio: np.ndarray,
    session_dir: Path,
    channel: str,
) -> tuple[str, str | None]:
    """Dispatch on the registry. Returns (text, error). Any exception is
    caught so one broken runtime doesn't kill the rest of the matrix."""
    cfg = MODEL_REGISTRY[model_key]
    try:
        if cfg["kind"] == "whisper":
            return run_faster_whisper(audio, cfg["fw_id"]), None
        elif cfg["kind"] == "nemo":
            return run_nemo_asr(
                audio,
                cfg["nemo_id"],
                extra_transcribe_kwargs=cfg.get("nemo_transcribe_kwargs"),
            ), None
        elif cfg["kind"] == "openoats_live":
            return run_openoats_live(session_dir, channel), None
        else:
            raise ValueError(f"Unknown kind: {cfg['kind']}")
    except Exception as exc:
        return "", f"{type(exc).__name__}: {exc}"


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--session",
        type=Path,
        required=True,
        help="OpenOats session directory (contains audio/mic.caf + audio/sys.caf).",
    )
    ap.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_REGISTRY.keys()),
        choices=list(MODEL_REGISTRY.keys()),
        help="Which models to run. Default: all.",
    )
    ap.add_argument(
        "--channels",
        nargs="+",
        default=["mic", "sys"],
        choices=["mic", "sys"],
        help="Which channels to transcribe. Default: both.",
    )
    ap.add_argument(
        "--run-name",
        default=time.strftime("%Y-%m-%d_%H-%M-%S"),
        help="Sub-directory under runs/ where outputs are written.",
    )
    args = ap.parse_args()

    session_dir: Path = args.session.expanduser().resolve()
    audio_dir = session_dir / "audio"
    if not audio_dir.is_dir():
        console.print(f"[red]No audio/ under {session_dir}[/red]")
        return 1

    run_dir = Path(__file__).resolve().parent / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Pre-load both channels once so we amortise the read cost.
    channel_audio: dict[str, np.ndarray] = {}
    for ch in args.channels:
        caf = audio_dir / f"{ch}.caf"
        if not caf.exists():
            console.print(f"[yellow]Skipping {ch}: {caf} missing[/yellow]")
            continue
        console.print(f"[cyan]Loading[/cyan] {ch}.caf ({caf.stat().st_size / 1e6:.1f} MB)…")
        channel_audio[ch] = load_mono_16k(caf)
        console.print(f"  → {len(channel_audio[ch]) / TARGET_SR:.1f}s at 16 kHz mono")

    results: list[ModelRun] = []

    for model_key in args.models:
        for ch, audio in channel_audio.items():
            model_dir = run_dir / model_key
            model_dir.mkdir(exist_ok=True)
            out_path = model_dir / f"{ch}.txt"
            metrics_path = model_dir / f"{ch}.metrics.json"

            console.print(f"\n[bold magenta]▶ {model_key} / {ch}[/bold magenta]")
            start = time.perf_counter()
            rss_before = peak_rss_mb()
            text, err = run_model(model_key, audio, session_dir, ch)
            wall = time.perf_counter() - start
            rss_after = peak_rss_mb()

            audio_seconds = len(audio) / TARGET_SR
            run = ModelRun(
                model_id=model_key,
                channel=ch,
                audio_seconds=audio_seconds,
                wall_seconds=wall,
                rtf=wall / audio_seconds if audio_seconds > 0 else 0.0,
                peak_rss_mb=max(rss_before, rss_after),
                text=text,
                error=err,
            )
            results.append(run)

            out_path.write_text(text or "", encoding="utf-8")
            metrics_path.write_text(json.dumps(run.to_json(), indent=2, ensure_ascii=False), encoding="utf-8")

            if err:
                console.print(f"[red]  {err}[/red]")
            else:
                console.print(
                    f"  [green]done[/green] in {wall:.1f}s  "
                    f"(RTF {run.rtf:.2f} · "
                    f"{len(text.split())} words · "
                    f"peak RSS {run.peak_rss_mb:.0f} MB)"
                )

    # Summary table
    console.print()
    table = Table(title=f"Bench summary — run {args.run_name}")
    table.add_column("Model")
    table.add_column("Channel")
    table.add_column("Audio", justify="right")
    table.add_column("Wall", justify="right")
    table.add_column("RTF", justify="right")
    table.add_column("Words", justify="right")
    table.add_column("Peak RSS", justify="right")
    table.add_column("Error")
    for r in results:
        table.add_row(
            r.model_id,
            r.channel,
            f"{r.audio_seconds:.0f}s",
            f"{r.wall_seconds:.1f}s",
            f"{r.rtf:.2f}",
            str(len((r.text or "").split())),
            f"{r.peak_rss_mb:.0f} MB",
            r.error or "",
        )
    console.print(table)

    summary_path = run_dir / "summary.json"
    summary_path.write_text(
        json.dumps([r.to_json() for r in results], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    console.print(f"\nOutputs in [bold]{run_dir}[/bold]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
