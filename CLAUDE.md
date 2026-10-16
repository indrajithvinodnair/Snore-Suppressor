# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Python 3.12 venv managed by uv (no pyproject.toml — install directly):

```bash
uv venv .venv --python 3.12
uv pip install numpy scipy matplotlib soundfile
```

Always run Python via `.venv/bin/python`, never `python3` or `python`.

## Commands

```bash
# Run all 8 synthetic scenarios + automated checks + regenerate dashboard
.venv/bin/python run.py --all

# Run a single scenario (1–8)
.venv/bin/python run.py --scenario 3

# Run on a real WAV file
.venv/bin/python run.py --audio samples/real-friend-snoring-extended.wav

# Regenerate dashboard.html from existing output directories (no reprocessing)
.venv/bin/python run.py --dashboard

# Run automated ground-truth checks (exit 0 = all pass)
.venv/bin/python tests/check_scenarios.py

# Serve dashboard
python3 -m http.server 8765 --directory outputs
```

## Architecture

The core pipeline separates concerns across three layers:

**Event stream** (`snore/generator.py`) — the canonical data format is a dict `{"duration", "background_amp", "pulses": [{"onset", "duration", "amplitude"}]}`. The detector operates on this abstract representation, not raw audio samples. `render_to_audio` converts a stream to float32 for WAV output and plotting only.

**Detection** (`snore/extractor.py` → `snore/detector.py`) — two stages:
1. `extract_events(audio, sr)` converts raw audio to an event stream. It auto-selects a quiet or noisy path based on the 10th-percentile RMS floor. The noisy path bandpass-filters (100–2000 Hz) before thresholding and merges fragments within 0.5 s.
2. `SnoreDetector` is a two-state FSM (`LISTENING → CANCELLING → LISTENING`). It locks a pattern after `patience` pulses whose inter-onset interval CV ≤ 0.30. Each subsequent pulse is matched against learned avg interval/duration/amplitude within `tolerance`; mismatches or timeouts break back to `LISTENING`. Running averages update on each confirmed cancellation with α = 0.10.

**Cancellation** (`snore/extractor.py`) — `subtract_cancelled_pulses` dispatches to one of three strategies:
- *Exact*: RNG-replay of the synthetic generator (zero residual, used for scenario mode)
- *Template* (`_template_subtract`): normalised cross-correlation to align stored raw-waveform templates from the learning phase, then least-squares optimal scale, then subtract with 20 ms cosine fades. Used for real audio.
- *Model*: fresh synthesised pulse per event — poor quality, fallback only.

**CLI / harness** (`run.py`) — ties everything together. `--mode stream` feeds the generator event list directly to the detector (no extractor). `--mode audio` (default for `--all`) renders to samples first, runs `extract_events`, then detects. For real audio (`--audio`), tolerance is auto-widened to 0.6 when background is noisy. Outputs go to `outputs/scenario_N_audio/` or `outputs/audio_<name>/`.

**Tests** (`tests/check_scenarios.py`) — compares `SnoreDetector.event_log` against ground-truth dicts from `generate_scenario`. Checks: should-lock, break timing within tolerance, jitter-must-not-break. Does not touch audio — runs on event streams only.

## Key parameters

| Parameter | Default | Where used |
|-----------|---------|------------|
| `patience` | 3 | Pulses to lock a pattern |
| `tolerance` | 0.25 (0.6 for noisy audio) | Max fractional deviation on interval/duration/amplitude |
| `_MAX_INTERVAL_CV` | 0.30 | Rejects irregular rhythms from locking |
| `chunk_ms` | 10 | RMS chunk size in `extract_events` |
| `_OFFSET_STREAK` | 3 | Consecutive sub-threshold chunks to declare offset (30 ms debounce) |
| `_MIN_DURATION_S` | 0.15 | Discard pulses shorter than this |
