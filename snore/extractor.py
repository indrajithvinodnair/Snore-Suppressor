"""
Pulse extractor: converts a float32 audio array to an event stream.
Bridges real audio → event-list representation for SnoreDetector.

Output event stream has the same schema as generator.generate_scenario():
  {"duration", "background_amp", "pulses": [{"onset","duration","amplitude"}, ...]}
"""

import numpy as np
from collections import deque
from scipy.signal import butter, sosfilt, sosfiltfilt

_OFFSET_STREAK  = 3     # consecutive sub-threshold chunks before pulse ends (3 × chunk_ms)
_MIN_DURATION_S = 0.15  # shorter than this is a noise transient — discard


def extract_events(audio: np.ndarray, sr: int = 22050, chunk_ms: int = 10) -> dict:
    """
    Scan audio for pulses; return an event stream ready for SnoreDetector.process().

    For quiet rooms, uses an adaptive rolling-max threshold (original approach).
    For recordings with significant background noise (bg > 0.005 RMS), first
    bandpass-filters 100–2000 Hz to suppress broadband noise while preserving snore
    frequencies, then applies a fixed threshold above the filtered noise floor.
    """
    chunk_size = int(sr * chunk_ms / 1000)
    n_chunks   = len(audio) // chunk_size
    if n_chunks == 0:
        return {"duration": len(audio) / sr, "background_amp": 0.0, "pulses": []}

    raw_rms = np.array([
        float(np.sqrt(np.mean(audio[i*chunk_size:(i+1)*chunk_size].astype(np.float64)**2)))
        for i in range(n_chunks)
    ])
    bg_floor = float(np.percentile(raw_rms, 10))

    # Significant background: bandpass-filter for detection to boost SNR.
    # Brown noise drops ~56 % in amplitude through 100–2000 Hz; snore is largely preserved.
    if bg_floor >= 0.005:
        nyq = sr / 2.0
        sos = butter(4, [100.0 / nyq, min(2000.0 / nyq, 0.99)], btype='band', output='sos')
        det_audio = sosfiltfilt(sos, audio.astype(np.float64)).astype(np.float32)
        det_rms   = np.array([
            float(np.sqrt(np.mean(det_audio[i*chunk_size:(i+1)*chunk_size].astype(np.float64)**2)))
            for i in range(n_chunks)
        ])
        # fixed threshold above the filtered noise floor
        fixed_thresh = max(0.003, float(np.percentile(det_rms, 25)) * 1.5)
    else:
        det_rms      = raw_rms
        fixed_thresh = None

    rms_window   = deque(maxlen=500)
    in_pulse     = False
    onset_idx    = 0
    quiet_streak = 0
    quiet_rms    = []
    pulses       = []

    for i in range(n_chunks):
        drms = det_rms[i]
        rms_window.append(drms)
        if fixed_thresh is not None:
            threshold = fixed_thresh
        else:
            threshold = max(0.01, max(rms_window) * 0.15) if rms_window else 0.01

        if drms > threshold:
            quiet_streak = 0
            if not in_pulse:
                in_pulse, onset_idx = True, i
        elif in_pulse:
            quiet_streak += 1
            if quiet_streak >= _OFFSET_STREAK:
                onset_s  = onset_idx * chunk_ms / 1000
                offset_s = (i + 1) * chunk_ms / 1000
                duration = offset_s - onset_s
                if duration >= _MIN_DURATION_S:
                    pulses.append({
                        "onset":     onset_s,
                        "duration":  duration,
                        "amplitude": float(raw_rms[onset_idx:i + 1].max()),
                    })
                in_pulse, quiet_streak = False, 0
        else:
            quiet_rms.append(drms)

    # In noisy backgrounds the snore amplitude fluctuates within a pulse; merge
    # fragments that are less than 0.5 s apart into a single pulse.
    if fixed_thresh is not None and len(pulses) > 1:
        merged = [dict(pulses[0])]
        for p in pulses[1:]:
            prev = merged[-1]
            if p["onset"] < prev["onset"] + prev["duration"] + 0.5:
                new_end = max(prev["onset"] + prev["duration"], p["onset"] + p["duration"])
                prev["duration"]  = new_end - prev["onset"]
                prev["amplitude"] = max(prev["amplitude"], p["amplitude"])
            else:
                merged.append(dict(p))
        pulses = merged

    if bg_floor >= 0.005:
        bg_amp = bg_floor
    else:
        bg_amp = float(np.median(quiet_rms)) if quiet_rms else 0.0

    return {"duration": len(audio) / sr, "background_amp": bg_amp, "pulses": pulses}


def subtract_cancelled_pulses(audio: np.ndarray, cancelled_pulses: list, sr: int = 22050,
                               original_stream: dict = None, seed: int = 99,
                               profile: dict = None) -> np.ndarray:
    """
    ANC-style cancellation: reconstruct each cancelled pulse waveform and subtract it.

    original_stream: exact generator stream → lossless RNG replay (synthetic audio).
    profile: learned snore profile from learn_snore_profile() → template subtraction (real audio).
    Neither: falls back to model-based subtraction (poor, kept as last resort).
    """
    out = audio.copy()
    if not cancelled_pulses:
        return out
    if original_stream is not None:
        _exact_subtract(out, cancelled_pulses, original_stream, sr, seed)
    elif profile is not None:
        _template_subtract(out, cancelled_pulses, sr, profile)
    else:
        _model_subtract(out, cancelled_pulses, sr)
    return np.clip(out, -1.0, 1.0)


def _exact_subtract(out: np.ndarray, cancelled_pulses: list, original_stream: dict,
                    sr: int, seed: int) -> None:
    """Replay RNG from seed to reconstruct each original pulse exactly; subtract cancelled ones."""
    from snore.generator import _render_pulse_audio
    n_samples = int(original_stream["duration"] * sr)
    rng = np.random.default_rng(seed)
    if original_stream["background_amp"] > 0:
        rng.standard_normal(n_samples)  # advance past background draw

    cancelled_onsets = {p["onset"] for p in cancelled_pulses}

    for p in original_stream["pulses"]:
        start, pulse = _render_pulse_audio(p["onset"], p["duration"], p["amplitude"],
                                           sr=sr, rng=rng)
        if any(abs(p["onset"] - co) < 0.15 for co in cancelled_onsets):
            end = min(start + len(pulse), len(out))
            out[start:end] -= pulse[:end - start]


def _model_subtract(out: np.ndarray, cancelled_pulses: list, sr: int) -> None:
    """Synthesize a model pulse per cancelled event and subtract it (real-audio path)."""
    from snore.generator import _render_pulse_audio
    for i, p in enumerate(cancelled_pulses):
        rng = np.random.default_rng(42 + i)
        start, pulse = _render_pulse_audio(p["onset"], p["duration"], p["amplitude"],
                                           sr=sr, rng=rng)
        end = min(start + len(pulse), len(out))
        out[start:end] -= pulse[:end - start]


def _template_subtract(out: np.ndarray, cancelled_pulses: list, sr: int, profile: dict) -> None:
    """
    For each cancelled pulse: find the best-matching raw learning-phase waveform via
    normalised cross-correlation, scale it to the detected amplitude, align it, subtract.
    """
    from scipy.signal import correlate
    raw_templates = profile["raw_templates"]
    raw_amps      = profile["raw_amps"]
    post_margin   = profile.get("max_pre_pad", int(sr * 0.05))
    bg_amp        = profile.get("bg_amp", 0.0)

    for p in cancelled_pulses:
        s = int(p["onset"] * sr)
        e = min(int((p["onset"] + p["duration"]) * sr), len(out))
        if e <= s:
            continue

        # Walk back to true acoustic onset; keep the larger window (never shrink vs max_pre_pad)
        true_s   = _find_true_onset(out, s, sr, p["amplitude"], bg_amp=bg_amp)
        search_s = min(true_s, max(0, s - post_margin))
        search_e = min(len(out), e + post_margin)
        search   = out[search_s:search_e]

        best_offset, best_tmpl, best_amp, best_score = 0, None, 1.0, -1.0

        for tmpl, t_amp in zip(raw_templates, raw_amps):
            t_len = min(len(tmpl), len(search))
            if t_len < 32:
                continue
            t = tmpl[:t_len]
            t_energy = float(np.dot(t, t))
            if t_energy == 0:
                continue

            corr = correlate(search, t, mode="valid")   # length: len(search)-t_len+1

            # sliding window energy of search for per-lag normalisation
            s_sq     = np.concatenate([[0.0], np.cumsum(search.astype(np.float64) ** 2)])
            s_energy = s_sq[t_len:] - s_sq[:-t_len]
            denom    = np.sqrt(t_energy * np.maximum(s_energy, 1e-12))
            norm_corr = np.abs(corr) / denom

            idx   = int(np.argmax(norm_corr))
            score = float(norm_corr[idx])
            if score > best_score:
                best_score, best_offset, best_tmpl, best_amp = score, idx, t, t_amp

        if best_tmpl is None:
            continue

        sub_s   = search_s + best_offset
        sub_e   = min(sub_s + len(best_tmpl), len(out))
        sub_len = sub_e - sub_s
        if sub_len <= 0:
            continue

        # Least-squares optimal scale: minimises residual energy for this alignment.
        # More accurate than peak_rms ratio, which drifts when adaptive threshold fires
        # at different points in the waveform for different pulses.
        t64          = best_tmpl[:sub_len].astype(np.float64)
        tmpl_energy  = float(np.dot(t64, t64))
        aligned      = out[sub_s:sub_e].astype(np.float64)
        if tmpl_energy > 0:
            scale = float(np.dot(t64, aligned)) / tmpl_energy
        else:
            scale = p["amplitude"] / best_amp if best_amp > 0 else 1.0

        scaled = (best_tmpl[:sub_len] * scale).astype(np.float32)

        # half-cosine fade in/out
        fade_n = min(int(sr * 0.020), len(scaled) // 4)
        t = np.arange(fade_n, dtype=np.float32)
        ramp = (0.5 * (1.0 - np.cos(np.pi * t / fade_n))).astype(np.float32)
        scaled[:fade_n]  *= ramp
        scaled[-fade_n:] *= ramp[::-1]

        out[sub_s:sub_e] -= scaled


def _find_true_onset(audio: np.ndarray, detected_s: int, sr: int, pulse_amp: float,
                     bg_amp: float = 0.0) -> int:
    """
    Walk backward from detected onset to the true acoustic start.

    With bg_amp: threshold = 5×background — adapts to each room's noise floor and
    correctly reaches the perceptible onset ~300-400 ms before the detector fires
    (the detector's adaptive threshold inflates after several snores, so it fires
    late into the rising edge).
    Without bg_amp: falls back to 5% of pulse peak (short, conservative walkback).
    """
    step   = max(1, int(sr * 0.005))             # 5 ms steps
    thresh = bg_amp * 5.0 if bg_amp > 0 else pulse_amp * 0.05
    limit  = max(0, detected_s - int(sr * 2.0))  # up to 2 s lookback
    pos    = detected_s
    while pos > limit:
        new_pos = max(limit, pos - step)
        rms = float(np.sqrt(np.mean(audio[new_pos:pos].astype(np.float64) ** 2)))
        if rms <= thresh:
            break
        pos = new_pos
    return pos


def _find_true_offset(audio: np.ndarray, detected_e: int, sr: int, pulse_amp: float,
                      bg_amp: float = 0.0) -> int:
    """Walk forward from detected offset to the true acoustic tail end (5×background threshold)."""
    step   = max(1, int(sr * 0.005))
    thresh = bg_amp * 5.0 if bg_amp > 0 else pulse_amp * 0.05
    limit  = min(len(audio), detected_e + int(sr * 2.0))
    pos    = detected_e
    while pos < limit:
        new_pos = min(limit, pos + step)
        rms = float(np.sqrt(np.mean(audio[pos:new_pos].astype(np.float64) ** 2)))
        if rms <= thresh:
            break
        pos = new_pos
    return pos


def learn_snore_profile(audio: np.ndarray, kept_pulses: list, sr: int = 22050,
                        bg_amp: float = 0.0) -> dict | None:
    """
    From learning-phase pulse windows, estimate the snore's dominant frequency band
    and build a normalized averaged waveform template.

    kept_pulses: pulses the detector did NOT cancel (learning phase).
    Returns None when there is nothing to learn from.
    """
    if not kept_pulses:
        return None

    # --- estimate dominant frequency band from each learning pulse ---
    lo_est, hi_est = [], []
    for p in kept_pulses:
        s = int(p["onset"] * sr)
        e = min(int((p["onset"] + p["duration"]) * sr), len(audio))
        chunk = audio[s:e].astype(np.float64)
        if len(chunk) < 32:
            continue
        power = np.abs(np.fft.rfft(chunk)) ** 2
        freqs = np.fft.rfftfreq(len(chunk), 1.0 / sr)
        total = power.sum()
        if total == 0:
            continue
        cum = np.cumsum(power) / total
        lo_est.append(float(freqs[np.searchsorted(cum, 0.10)]))
        hi_est.append(float(freqs[np.searchsorted(cum, 0.90)]))

    if not lo_est:
        return None

    lo_hz = max(float(np.median(lo_est)), 50.0)
    hi_hz = min(float(np.median(hi_est)), sr / 2.0 - 10.0)
    if hi_hz <= lo_hz:
        hi_hz = lo_hz * 3.0

    # --- bandpass-filter each learning pulse and average → template ---
    nyq = sr / 2.0
    sos = butter(4, [lo_hz / nyq, hi_hz / nyq], btype="band", output="sos")

    templates, amps = [], []
    for p in kept_pulses:
        s = int(p["onset"] * sr)
        e = min(int((p["onset"] + p["duration"]) * sr), len(audio))
        chunk = audio[s:e].astype(np.float32)
        if len(chunk) < 32:
            continue
        filtered = sosfilt(sos, chunk).astype(np.float32)
        peak = float(np.max(np.abs(filtered)))
        if peak > 0:
            templates.append(filtered / peak)
            amps.append(p["amplitude"])

    if not templates:
        return None

    max_len = max(len(t) for t in templates)
    padded = np.array([np.pad(t, (0, max_len - len(t))) for t in templates])
    template = padded.mean(axis=0).astype(np.float32)

    # Raw (unfiltered) waveforms — extended to true acoustic onset/offset
    # Also measure RMS envelope rise/decay times per pulse for diagnostics
    raw_templates, raw_amps, pre_pads = [], [], []
    rise_times_s, decay_times_s = [], []
    env_chunk = max(1, int(sr * 0.005))  # 5ms chunks for envelope

    for p in kept_pulses:
        det_s = int(p["onset"] * sr)
        det_e = min(int((p["onset"] + p["duration"]) * sr), len(audio))
        true_s = _find_true_onset(audio, det_s, sr, p["amplitude"], bg_amp=bg_amp)
        true_e = _find_true_offset(audio, det_e, sr, p["amplitude"], bg_amp=bg_amp)
        chunk = audio[true_s:true_e].astype(np.float32)
        if len(chunk) >= 32:
            raw_templates.append(chunk)
            raw_amps.append(p["amplitude"])
            pre_pads.append(det_s - true_s)

        # Envelope: work on detected window to measure shape relative to threshold
        det_chunk = audio[det_s:det_e]
        n_env = len(det_chunk) // env_chunk
        if n_env >= 3:
            env = np.array([
                float(np.sqrt(np.mean(det_chunk[i*env_chunk:(i+1)*env_chunk].astype(np.float64)**2)))
                for i in range(n_env)
            ])
            peak_idx = int(np.argmax(env))
            rise_times_s.append(peak_idx * env_chunk / sr)
            decay_times_s.append((n_env - 1 - peak_idx) * env_chunk / sr)

    avg_rise_time_s  = float(np.mean(rise_times_s))  if rise_times_s  else 0.0
    avg_decay_time_s = float(np.mean(decay_times_s)) if decay_times_s else 0.0

    return {
        "lo_hz":            lo_hz,
        "hi_hz":            hi_hz,
        "template":         template,
        "template_amp":     float(np.mean(amps)),
        "raw_templates":    raw_templates,
        "raw_amps":         raw_amps,
        "max_pre_pad":      max(pre_pads) if pre_pads else int(sr * 0.05),
        "avg_rise_time_s":  avg_rise_time_s,
        "avg_decay_time_s": avg_decay_time_s,
        "bg_amp":           bg_amp,
    }


