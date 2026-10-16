"""
Automated pass/fail checks for all 8 scenarios.
Compares detector event_log against known ground truth — no listening needed.
Exit code 0 iff all pass.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from snore.detector import SnoreDetector
from snore.generator import generate_scenario, SR


def _events_of_type(log, event_type):
    return [e for e in log if e["event"] == event_type]


def _first_event(log, event_type):
    matches = _events_of_type(log, event_type)
    return matches[0] if matches else None


def _run_scenario(n, patience=3, tolerance=0.25, strength=1.0):
    stream, gt = generate_scenario(n)
    det = SnoreDetector(sr=SR, patience=patience, tolerance=tolerance, strength=strength)
    det.process(stream)
    return det.event_log, gt


def _check(n, patience=3, tolerance=0.25, strength=1.0):
    log, gt = _run_scenario(n, patience, tolerance, strength)
    learned = _first_event(log, "learned")
    breaks = _events_of_type(log, "break")

    # --- should-lock check ---
    if gt["should_lock"] and learned is None:
        return False, "expected detector to lock into CANCELLING but it never learned a pattern"
    if not gt["should_lock"] and learned is not None:
        return False, f"expected no lock but learned at t={learned['time']:.2f}s"

    # --- break check (scenarios 3, 4, 5) ---
    if gt["break_after"] is not None:
        first_break = _first_event(log, "break")
        if first_break is None:
            return False, "expected a pattern break but none was logged"

        too_early = first_break["time"] < gt["break_after"]
        deadline = gt["break_after"] + gt["break_tolerance_s"]
        too_late = first_break["time"] > deadline

        if too_early:
            return False, (f"break at t={first_break['time']:.2f}s is before the authored "
                           f"change at t={gt['break_after']:.2f}s")
        if too_late:
            return False, (f"break at t={first_break['time']:.2f}s is more than "
                           f"{gt['break_tolerance_s']:.1f}s after the change at "
                           f"t={gt['break_after']:.2f}s")

    # --- jitter must not break (scenario 2) ---
    if n == 2 and breaks:
        return False, f"jitter scenario should not break pattern but got break at t={breaks[0]['time']:.2f}s"

    return True, "ok"


SCENARIO_NAMES = {
    1: "Steady rhythm",
    2: "Steady + jitter",
    3: "Rhythm change at 6 s",
    4: "Amplitude change at 7.5 s",
    5: "Duration change at 7.5 s",
    6: "Irregular pulses",
    7: "Background noise only",
    8: "Silence",
}


def run_all_checks(patience=3, tolerance=0.25, strength=1.0):
    all_passed = True
    for n in range(1, 9):
        passed, reason = _check(n, patience, tolerance, strength)
        status = "PASS" if passed else "FAIL"
        label = SCENARIO_NAMES[n]
        if passed:
            print(f"  [{status}] Scenario {n}: {label}")
        else:
            print(f"  [{status}] Scenario {n}: {label} — {reason}")
        if not passed:
            all_passed = False
    return all_passed


if __name__ == "__main__":
    print("Running automated checks...\n")
    ok = run_all_checks()
    print()
    if ok:
        print("All checks passed.")
        sys.exit(0)
    else:
        print("Some checks failed.")
        sys.exit(1)
