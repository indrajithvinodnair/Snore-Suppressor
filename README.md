# 🔇 Snore Suppressor

Rule-based snore detection and ANC-style cancellation. No ML.

**[Live dashboard →]([https://indrajithvinodnair.github.io/snore-suppressor/](https://indrajithvinodnair.github.io/Snore-Suppressor/))**

---

## What is this?

A Python PoC tackling a real-world problem of my own. I share a room with a friend who snores, and I've tried both music and noise-cancelling earphones to get some sleep. Music kept me awake, while the earphones couldn't fully eliminate the snoring.

Since neither solution worked for me, I decided to build something of my own. The idea is to detect rhythmic snoring in audio, learn its pattern, and cancel matching pulses using template subtraction, inspired by the principles behind active noise cancellation.

The entire approach is rule-based, with no training data, classifier, or pretrained model.
---

## The problem

Snoring is periodic. Same interval, same duration, same amplitude — cycle after cycle after cycle. That regularity is both the detection signal and the cancellation handle. The goal is to leave the background unchanged and suppress only the snore pulses.

---

## Why not ML?

Most snore-detection work trains classifiers on spectrograms or MFCCs. This project takes a different position: **snoring doesn't need a model — it needs a ruler.**

- No training data required
- No false-positive/negative tuning across recording conditions
- The rhythm itself is the detector; once locked, the pattern predicts the next pulse *before it arrives*
- The detector operates on an event stream (onset, duration, amplitude tuples), not raw audio samples — a design insight from the project's own development: *"No need to operate on a sound wave at all if required. Only a mathematical representation of the soundwave is a good thing to be used as input and used for processing."*

---

## How detection works

Two-stage pipeline: `snore/extractor.py` → `snore/detector.py`.

### Stage 1 — Pulse extraction (`extract_events`)

1. Slice audio into 10 ms chunks; compute RMS per chunk.
2. Measure background floor as the 10th-percentile of RMS values.
3. Choose detection path based on floor amplitude:
   - **Quiet path** (floor < 0.005): adaptive threshold = 15% of the rolling-max RMS over the last 5 s.
   - **Noisy path** (floor ≥ 0.005): bandpass-filter 100–2000 Hz first — this suppresses roughly 56% of brown-noise amplitude while preserving snore energy. Fixed threshold = max(0.003, P25 × 1.5). Merge pulse fragments within 0.5 s.
4. Rising edge = onset. Three consecutive sub-threshold chunks (30 ms debounce) = offset. Pulses shorter than 150 ms are discarded as transients.

### Stage 2 — Rhythm learning (`SnoreDetector`)

The detector is a two-state FSM: `LISTENING → CANCELLING → LISTENING`.

After `patience` consecutive pulses (default 3), compute the inter-onset interval coefficient of variation (CV = σ/μ):

- **CV > 0.30**: rhythm too irregular — slide the window forward one pulse and try again. Bimodal gaps, random transients, and irregular bursts never lock.
- **CV ≤ 0.30**: lock into `CANCELLING`. Record average interval, duration, and amplitude.

While cancelling, each arriving pulse is checked against all three learned averages. A match (all within `tolerance`, default 25%) → cancel. A mismatch, or a timeout (silence longer than expected × 1.25) → break back to `LISTENING`.

Matched cancellations update all three running averages with α = 0.10 exponential smoothing, allowing natural drift without triggering unnecessary breaks.

---

## How cancellation works

The ANC analogy: headphones capture the noise waveform, invert it, and play the anti-noise simultaneously — `noise + (−noise) = 0`. Snoring is actually *easier* to cancel than random ambient noise, because it's rhythmic. We can predict when the next pulse arrives before it starts; ANC headphones have to react to unpredictable noise in real time.

A key insight from the design process: *"The detection is using the summary to learn the rhythm pattern. But the cancellation process needs to learn from the raw waveform from the detected snore pulses to pin-point the counter for the snore pulses."* The detector learns rhythm from event tuples; cancellation learns the waveform shape from the raw audio.

### Template subtraction (`_template_subtract` in `snore/extractor.py`)

1. During the `patience` learning phase, store the raw waveform window for each kept pulse — bandpass-filtered, peak-normalised, and averaged across the `patience` pulses to form a reference template.
2. For each cancelled pulse: normalised cross-correlation against all stored templates (scipy `correlate`, per-lag energy normalisation from a cumulative sum-of-squares) finds the best alignment offset across all candidates.
3. Least-squares optimal scale: `scale = dot(T, S) / dot(T, T)` — minimises residual energy after subtraction; more accurate than a peak-RMS ratio when the detection threshold fires at different rise-edge positions on different pulses.
4. Subtract the scaled, aligned template with 20 ms half-cosine fades at both ends to avoid click artefacts.
5. Result: `(snore + background) − snore ≈ background`.

---

## The challenges

### 1 — Late onset detection (the "seam")

The RMS threshold fires at ~15% of peak amplitude. The quiet rising edge of each snore is missed, so the detected onset is 400–500 ms after the acoustic onset. Template subtraction compensates by walking back from the detected onset at 5 ms steps (envelope walk until RMS < 5 × bg_amp), but when the walk-back margin found during learning is shorter than the true acoustic onset of a cancelled pulse, a ~0.5 s seam remains audible at the start of each cancellation.

### 2 — Cut-and-paste artefacts (early approach, replaced)

An early version overwrote cancelled windows with background-noise samples spliced in from elsewhere. This produced an audible "whip whip" artefact — a gap where the snore was, bracketed by two splice seams. The right model is subtraction, not replacement: *"The approach of replacing the sample range of cancelled pulse with background audio drawn from elsewhere is bad. Ideally we should target only the snore pulse and remove it. Remember even when the snore pulse is occurring there is underlying background noise that's overshadowed by the towering snore pulse."* Template subtraction replaced the cut-and-paste approach entirely.

### 3 — Amplitude scaling drift across nominally identical pulses

Because the adaptive threshold fires at different rise-edge positions, detected windows vary slightly from pulse to pulse, making `peak_rms` estimates drift by ±10%. This causes the least-squares scale to over- or under-subtract — explaining why some pulses in a recording of repeated identical snores cancel more cleanly than others.

### 4 — Template contamination in noisy recordings

Template subtraction is an ANC technique — it works when the reference is a *clean* copy of the interference. Here the learned template captures `snore + noise_A`. The pulse to cancel is `snore + noise_B`. After subtraction: `noise_B − noise_A` — a new noise artefact. The least-squares scale is also biased: `scale ≈ snore_energy / (snore_energy + noise_energy) ≈ 0.58` at typical SNRs, so only ~58% of the snore is removed. For recordings with significant background noise, a noise-isolation step before template learning (e.g. spectral gating) is needed before template subtraction makes sense.

---

## Test harness

`tests/check_scenarios.py` — 8 synthetic scenarios with known ground truth, all passing:

| # | Scenario | What it tests |
|---|----------|---------------|
| 1 | Steady rhythm | Locks and stays locked |
| 2 | Steady + jitter (≤ 8%) | Locks; jitter must not trigger a break |
| 3 | Rhythm doubles at 6 s | Locks before 6 s; rhythm-timeout break detected within tolerance |
| 4 | Amplitude drops 60% at 7.5 s | Cancels pulse at 6 s; amplitude break fires at 7.5 s |
| 5 | Duration doubles at 7.5 s | Cancels pulse at 6 s; duration break fires at 7.5 s |
| 6 | Bimodal gaps (alternating short/long) | CV always > 0.30; never locks |
| 7 | Background noise only | Never locks |
| 8 | Silence | Never locks |

Scenarios 3, 4, and 5 exercise the full `LISTENING → CANCELLING → break → LISTENING` lifecycle. Scenarios 6, 7, and 8 are negative controls for the lock-refusal logic.

Real audio results:

| File | Extracted | Cancelled |
|------|-----------|-----------|
| `samples/real-friend-snoring-extended.wav` (quiet room) | 9 | 6 |
| `samples/soft-brown_real-friend-snoring-exte_mixdown.wav` (brown noise) | 9 | 6 (quality limited by challenge 4) |

---

## Quick start

```bash
# Install deps (uv required)
uv sync

# Run all scenarios + automated checks + generate dashboard
.venv/bin/python run.py --all

# Run on real audio
.venv/bin/python run.py --audio samples/real-friend-snoring-extended.wav

# Serve and view dashboard
python3 -m http.server 8765 --directory outputs
# open http://localhost:8765/dashboard.html

# Run automated checks only
.venv/bin/python tests/check_scenarios.py
```

Key CLI flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--patience N` | 3 | Pulses required before locking a pattern |
| `--tolerance F` | 0.25 | Max fractional deviation (25%) on interval, duration, amplitude |
| `--mode stream\|audio` | stream | `stream` feeds event tuples directly; `audio` renders to samples first |

---

## Project structure

```
snore/
  generator.py        — synthetic event-stream scenarios; render_to_audio
  detector.py         — SnoreDetector FSM (LISTENING → CANCELLING → LISTENING)
  extractor.py        — extract_events; learn_snore_profile; subtract_cancelled_pulses
run.py                — CLI harness, dashboard generation, waveform plots
tests/
  check_scenarios.py  — automated ground-truth checks
samples/              — scenario WAVs + two real recordings
outputs/              — generated WAVs, waveform PNGs, dashboard.html
```

---

## Known limits

- **Noisy recordings**: template contamination (challenge 4) means cancellation quality degrades with background noise. A spectral gating or noise-estimation step before template learning would address this.
- **Onset latency**: the ~0.5 s seam at the start of each cancellation (challenge 1) could be reduced with a lower initial detection threshold or an envelope-shape model learned during the patience phase.
- **Offline only**: cancellation is batch/offline. Real-time operation would require a ring-buffer lookahead of `patience × avg_interval` seconds to accumulate the learning pulses before cancellation could begin.
