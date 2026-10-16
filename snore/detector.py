"""
SnoreDetector: event-list state machine.

Input:  event stream  {"duration", "background_amp", "pulses": [...]}
Output: processed stream — matched snore pulses removed, background always preserved.

States: LISTENING → CANCELLING → LISTENING (reset after a break).

Working on exact pulse events (onset, duration, amplitude) eliminates RMS
measurement noise and chunk-boundary artefacts from the earlier sample-based design.
"""

import numpy as np

LISTENING  = "listening"
CANCELLING = "cancelling"

# Shorter than this → treat as noise / transient, pass through without learning.
_MIN_PULSE_DURATION_S = 0.15

# Pattern lock refused when interval CV exceeds this (irregular rhythms have CV >> 0.5).
_MAX_INTERVAL_CV = 0.30


class SnoreDetector:
    def __init__(self, sr=22050, patience=3, tolerance=0.25, strength=1.0):
        self.sr        = sr          # kept for external callers; not used internally
        self.patience  = patience    # pulses needed to lock a pattern
        self.tolerance = tolerance   # fractional deviation allowed before break
        self.strength  = strength    # kept for API compat; cancellation is always full
        self.event_log = []
        self._reset_state()

    # ------------------------------------------------------------------ state

    def _reset_state(self):
        """Reset pattern state. event_log survives resets."""
        self.state         = LISTENING
        self._pulse_history = []
        self.avg_interval  = None
        self.avg_duration  = None
        self.avg_peak      = None

    # -------------------------------------------------------- public interface

    def process(self, stream: dict) -> dict:
        """
        Process event stream. Returns stream with matched snore pulses removed.
        Background noise (background_amp) is always preserved in the output.
        """
        self.event_log = []
        self._reset_state()

        pulses = sorted(stream["pulses"], key=lambda p: p["onset"])
        kept      = []
        cancelled = []

        for pulse in pulses:
            if not self._pulse_is_long_enough(pulse):
                kept.append(pulse)   # noise transient — pass through, skip learning
                continue

            if self.state == LISTENING:
                kept.append(pulse)
                self._observe_listening(pulse)
            else:  # CANCELLING
                if self._rhythm_break_before(pulse):
                    self._do_break(pulse["onset"], "rhythm-timeout")
                    kept.append(pulse)
                    self._observe_listening(pulse)
                elif self._pulse_matches_pattern(pulse):
                    cancelled.append(pulse)
                    self._update_pattern(pulse)
                    self._pulse_history.append(pulse)
                else:
                    reason = self._deviation_reason(pulse)
                    self._do_break(pulse["onset"], reason)
                    kept.append(pulse)
                    self._observe_listening(pulse)

        return {
            "duration":       stream["duration"],
            "background_amp": stream["background_amp"],
            "pulses":         kept,       # snore pulses that were NOT cancelled
            "cancelled":      cancelled,  # snore pulses that were removed
        }

    # -------------------------------------------------- listening helpers

    def _observe_listening(self, pulse):
        self._pulse_history.append(pulse)
        if len(self._pulse_history) >= self.patience:
            self._try_learn_pattern(pulse["onset"])

    def _try_learn_pattern(self, current_time: float):
        recent   = self._pulse_history[-self.patience:]
        onsets   = [p["onset"] for p in recent]
        intervals = [onsets[i+1] - onsets[i] for i in range(len(onsets) - 1)]
        cv = self._coefficient_of_variation(intervals)

        if self._rhythm_is_too_irregular(cv):
            # slide the window forward rather than discarding all history
            self._pulse_history.pop(0)
            return

        self.avg_interval = float(np.mean(intervals))
        self.avg_duration = float(np.mean([p["duration"]  for p in recent]))
        self.avg_peak     = float(np.mean([p["amplitude"] for p in recent]))

        self.event_log.append({"event": "learned", "time": current_time,
                                "after_pulses": len(self._pulse_history)})
        self.state = CANCELLING

    # -------------------------------------------------- cancelling helpers

    def _do_break(self, onset_s: float, reason: str):
        self.event_log.append({"event": "break", "time": onset_s, "reason": reason})
        self._reset_state()

    def _update_pattern(self, pulse):
        # exponential smoothing — lets pattern drift slowly with natural variation
        alpha = 0.1
        if self._pulse_history:
            gap = pulse["onset"] - self._pulse_history[-1]["onset"]
            self.avg_interval = (1 - alpha) * self.avg_interval + alpha * gap
        self.avg_duration = (1 - alpha) * self.avg_duration + alpha * pulse["duration"]
        self.avg_peak     = (1 - alpha) * self.avg_peak     + alpha * pulse["amplitude"]

    # ---------------------------------------------- intention-revealing checks

    def _rhythm_break_before(self, pulse) -> bool:
        """Gap from the last known pulse is too wide — snorer changed rhythm."""
        if not self._pulse_history:
            return False
        gap = pulse["onset"] - self._pulse_history[-1]["onset"]
        return gap > self.avg_interval * (1.0 + self.tolerance)

    def _pulse_matches_pattern(self, pulse) -> bool:
        interval_ok  = self._within_tolerance(self._interval_deviation(pulse))
        duration_ok  = self._within_tolerance(self._duration_deviation(pulse))
        amplitude_ok = self._within_tolerance(self._amplitude_deviation(pulse))
        return interval_ok and duration_ok and amplitude_ok

    def _rhythm_is_too_irregular(self, cv: float) -> bool:
        return cv > _MAX_INTERVAL_CV

    def _pulse_is_long_enough(self, pulse) -> bool:
        return pulse["duration"] >= _MIN_PULSE_DURATION_S

    def _within_tolerance(self, deviation: float) -> bool:
        return deviation <= self.tolerance

    # ------------------------------------------- deviation measurements

    def _interval_deviation(self, pulse) -> float:
        if not self._pulse_history or self.avg_interval is None:
            return 0.0
        gap = pulse["onset"] - self._pulse_history[-1]["onset"]
        return abs(gap - self.avg_interval) / max(self.avg_interval, 1e-6)

    def _duration_deviation(self, pulse) -> float:
        if self.avg_duration is None:
            return 0.0
        return abs(pulse["duration"] - self.avg_duration) / max(self.avg_duration, 1e-6)

    def _amplitude_deviation(self, pulse) -> float:
        if self.avg_peak is None:
            return 0.0
        return abs(pulse["amplitude"] - self.avg_peak) / max(self.avg_peak, 1e-6)

    def _deviation_reason(self, pulse) -> str:
        reasons = []
        if not self._within_tolerance(self._interval_deviation(pulse)):
            reasons.append("rhythm")
        if not self._within_tolerance(self._duration_deviation(pulse)):
            reasons.append("duration")
        if not self._within_tolerance(self._amplitude_deviation(pulse)):
            reasons.append("amplitude")
        return "+".join(reasons) if reasons else "unknown"

    # ------------------------------------------------- utilities

    @staticmethod
    def _coefficient_of_variation(values) -> float:
        arr  = np.array(values, dtype=float)
        mean = np.mean(arr)
        if mean < 1e-9:
            return 0.0
        return float(np.std(arr) / mean)
