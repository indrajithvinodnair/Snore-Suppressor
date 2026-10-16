"""
Synthetic snore event-stream generator.
Produces 8 labeled scenarios as event streams — mathematical pulse lists,
no audio samples. Background noise is a preserved parameter, not a pulse.

Event stream schema:
  {
    "duration":       float,   total duration in seconds
    "background_amp": float,   broadband noise floor amplitude — always kept
    "pulses": [
      {"onset": float, "duration": float, "amplitude": float},
      ...
    ]
  }
"""

import numpy as np
from scipy.signal import butter, sosfilt

SR = 22050
PULSE_LO_HZ = 100
PULSE_HI_HZ = 400


# -------------------------------------------------------- pulse list builder

def _make_pulse_list(duration, interval, dur, amp, rng=None, jitter_frac=0.0,
                     change_time=None, change_type=None,
                     changed_interval=None, changed_amp=None, changed_dur=None):
    if rng is None:
        rng = np.random.default_rng()
    pulses = []
    t = interval  # first onset one interval in
    while t < duration:
        after_change = change_time is not None and t >= change_time
        cur_interval = changed_interval if (after_change and change_type == "rhythm") else interval
        cur_amp      = changed_amp      if (after_change and change_type == "amplitude") else amp
        cur_dur      = changed_dur      if (after_change and change_type == "duration") else dur
        jitter = rng.uniform(-jitter_frac, jitter_frac) * cur_interval if jitter_frac else 0.0
        pulses.append({"onset": t + jitter, "duration": cur_dur, "amplitude": cur_amp})
        t += cur_interval
    return pulses


def _ground_truth(should_lock, changes=None, break_after=None, break_tolerance_s=3.0):
    return {
        "should_lock": should_lock,
        "changes": changes or [],
        "break_after": break_after,
        "break_tolerance_s": break_tolerance_s,
    }


# -------------------------------------------------------- scenario catalogue

def generate_scenario(n: int, duration=10.0, seed=42):
    """Return (event_stream, ground_truth_dict) for scenario n (1–8)."""
    rng = np.random.default_rng(seed)

    interval = 1.5   # seconds between onsets — 6 pulses fit in 10 s
    dur      = 0.4   # pulse duration seconds
    amp      = 0.4   # peak amplitude
    bg_amp   = 0.003 # background noise floor (always kept)

    if n == 1:
        # steady rhythm, no changes — should lock and stay locked
        pulses = _make_pulse_list(duration, interval, dur, amp)
        gt = _ground_truth(should_lock=True)

    elif n == 2:
        # steady + small jitter (≤ 8 %) — should lock, jitter must not break it
        pulses = _make_pulse_list(duration, interval, dur, amp, rng=rng, jitter_frac=0.08)
        gt = _ground_truth(should_lock=True)

    elif n == 3:
        # rhythm doubles after 6 s — should lock before 6 s, then break
        pulses = _make_pulse_list(duration, interval, dur, amp,
                                  change_time=6.0, change_type="rhythm",
                                  changed_interval=interval * 2.0)
        # pulse at t=6.0 is still on old rhythm and gets cancelled;
        # next pulse at t=9.0, gap=3.0 > 1.5*1.25=1.875 → timeout at t=9.0
        gt = _ground_truth(should_lock=True,
                           changes=[{"time": 6.0, "type": "rhythm"}],
                           break_after=6.0, break_tolerance_s=3.0)

    elif n == 4:
        # amplitude drops 60 % after 7.5 s — pulse at 6.0 s gets cancelled first,
        # then pulse at 7.5 s breaks the pattern (demonstrates lock → cancel → break)
        pulses = _make_pulse_list(duration, interval, dur, amp,
                                  change_time=7.5, change_type="amplitude",
                                  changed_amp=amp * 0.4)
        gt = _ground_truth(should_lock=True,
                           changes=[{"time": 7.5, "type": "amplitude"}],
                           break_after=7.5)

    elif n == 5:
        # duration doubles after 7.5 s — pulse at 6.0 s gets cancelled first,
        # then pulse at 7.5 s breaks the pattern (demonstrates lock → cancel → break)
        pulses = _make_pulse_list(duration, interval, dur, amp,
                                  change_time=7.5, change_type="duration",
                                  changed_dur=dur * 2.2)
        gt = _ground_truth(should_lock=True,
                           changes=[{"time": 7.5, "type": "duration"}],
                           break_after=7.5)

    elif n == 6:
        # bimodal gaps (short/long alternating) — CV always >> 0.30, never locks
        pulses = []
        t = 0.3
        use_short = True
        while t < duration - 0.5:
            pulses.append({"onset": t, "duration": float(rng.uniform(0.2, 0.4)),
                           "amplitude": amp})
            gap = rng.uniform(0.3, 0.6) if use_short else rng.uniform(2.0, 4.0)
            t += float(gap)
            use_short = not use_short
        gt = _ground_truth(should_lock=False)

    elif n == 7:
        # background noise only, no pulses — should never lock
        pulses = []
        gt = _ground_truth(should_lock=False)

    elif n == 8:
        # silence — no pulses, no background
        pulses = []
        bg_amp = 0.0
        gt = _ground_truth(should_lock=False)

    else:
        raise ValueError(f"Scenario must be 1–8, got {n}")

    return {"duration": duration, "background_amp": bg_amp, "pulses": pulses}, gt


# --------------------------------- audio rendering (for wav output and plots)

def _bandpass_sos(lo=PULSE_LO_HZ, hi=PULSE_HI_HZ, sr=SR, order=4):
    nyq = sr / 2
    return butter(order, [lo / nyq, hi / nyq], btype="band", output="sos")


def _render_pulse_audio(onset_s, duration_s, amplitude, sr=SR, rng=None):
    """Synthesize one bandpass noise pulse from event parameters."""
    if rng is None:
        rng = np.random.default_rng()
    n = int(duration_s * sr)
    noise = rng.standard_normal(n).astype(np.float32)
    sos = _bandpass_sos(sr=sr)
    filtered = sosfilt(sos, noise).astype(np.float32)
    envelope = np.hanning(n).astype(np.float32)
    pulse = filtered * envelope
    mx = np.max(np.abs(pulse))
    if mx > 0:
        pulse *= amplitude / mx
    return int(onset_s * sr), pulse


def render_to_audio(stream, sr=SR, seed=99):
    """Convert an event stream to a float32 audio array for listening / plotting."""
    rng = np.random.default_rng(seed)
    n_samples = int(stream["duration"] * sr)
    audio = np.zeros(n_samples, dtype=np.float32)

    if stream["background_amp"] > 0:
        audio += rng.standard_normal(n_samples).astype(np.float32) * stream["background_amp"]

    for p in stream["pulses"]:
        start, pulse = _render_pulse_audio(p["onset"], p["duration"], p["amplitude"],
                                           sr=sr, rng=rng)
        end = min(start + len(pulse), n_samples)
        audio[start:end] += pulse[:end - start]

    return np.clip(audio, -1.0, 1.0)
