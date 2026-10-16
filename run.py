"""
CLI entry point.

Usage:
  python run.py --scenario 1                        # stream mode (default)
  python run.py --scenario 1 --mode audio           # render → extract → detect
  python run.py --all
  python run.py --all --mode audio
  python run.py --audio samples/scenario_1.wav      # real WAV file
  python run.py --make-samples                      # write samples/ WAVs
  python run.py --scenario 2 --patience 4 --tolerance 0.20
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import soundfile as sf

from snore.detector import SnoreDetector, LISTENING, CANCELLING
from snore.generator import generate_scenario, render_to_audio, SR
from snore.extractor import extract_events, subtract_cancelled_pulses, learn_snore_profile
from tests.check_scenarios import run_all_checks, SCENARIO_NAMES


# ------------------------------------------------------------------ plot

STATE_COLORS = {
    LISTENING:  "#d0d0d0",  # grey
    CANCELLING: "#a8e6a0",  # green
    "broken":   "#f4a0a0",  # red
}


def _plot_waveform(audio, event_log, out_path, sr=SR, title="Waveform", cancelled=None):
    times = np.linspace(0, len(audio) / sr, len(audio))
    fig, ax = plt.subplots(figsize=(12, 3))

    segments = []
    state, seg_start = LISTENING, 0.0
    for ev in sorted(event_log, key=lambda e: e["time"]):
        if ev["event"] == "learned":
            segments.append((seg_start, ev["time"], state))
            seg_start, state = ev["time"], CANCELLING
        elif ev["event"] == "break":
            segments.append((seg_start, ev["time"], state))
            segments.append((ev["time"], ev["time"] + 0.3, "broken"))
            seg_start, state = ev["time"] + 0.3, LISTENING
    segments.append((seg_start, len(audio) / sr, state))

    for t0, t1, seg_state in segments:
        ax.axvspan(t0, t1, alpha=0.35, color=STATE_COLORS.get(seg_state, "#fff"), linewidth=0)

    ax.plot(times, audio, color="#333333", linewidth=0.4, rasterized=True)

    if cancelled:
        for p in cancelled:
            ax.axvspan(p["onset"], p["onset"] + p["duration"],
                       alpha=0.25, color="#ff6600", linewidth=0, hatch="//", label="_")

    ax.set_xlim(0, len(audio) / sr)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.legend(handles=[
        mpatches.Patch(color=STATE_COLORS[LISTENING],  alpha=0.6, label="Listening"),
        mpatches.Patch(color=STATE_COLORS[CANCELLING], alpha=0.6, label="Cancelling"),
        mpatches.Patch(color=STATE_COLORS["broken"],   alpha=0.6, label="Pattern break"),
    ], loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- summary

def _summary(event_log, orig_stream, result) -> str:
    learned = next((e for e in event_log if e["event"] == "learned"), None)
    breaks  = [e for e in event_log if e["event"] == "break"]

    parts = []
    if learned:
        parts.append(f"Learned after {learned['after_pulses']} pulses at {learned['time']:.1f}s")
    else:
        parts.append("Never locked into a pattern")

    if breaks:
        parts.append(f"detected break at {breaks[0]['time']:.1f}s ({breaks[0]['reason']})")
    elif learned:
        parts.append("no breaks detected")

    n_total     = len(orig_stream["pulses"])
    n_cancelled = len(result["cancelled"])
    parts.append(f"{n_total} extracted / {n_cancelled} cancelled")

    return ", ".join(parts) + "."


# --------------------------------------------------------- shared save helper

def _run_and_save(stream, audio, sr, out_dir, title, patience, tolerance, strength,
                  original_stream=None):
    """
    Run detector on stream; write outputs to out_dir.
    audio: original float32 array when in audio mode; None in stream mode.
    original_stream: exact generator stream for ANC exact subtraction (audio mode only).
    """
    os.makedirs(out_dir, exist_ok=True)

    det    = SnoreDetector(sr=sr, patience=patience, tolerance=tolerance, strength=strength)
    result = det.process(stream)

    if audio is not None:
        original_audio = audio
        if original_stream is not None:
            # synthetic audio: exact RNG replay → zero residual
            processed_audio = subtract_cancelled_pulses(audio, result["cancelled"], sr=sr,
                                                        original_stream=original_stream, seed=99)
        else:
            # real audio: learn frequency band + template from kept (learning) pulses
            cancelled_onsets = {p["onset"] for p in result["cancelled"]}
            kept = [p for p in stream["pulses"] if p["onset"] not in cancelled_onsets]
            profile = learn_snore_profile(audio, kept, sr, bg_amp=stream["background_amp"])
            processed_audio = subtract_cancelled_pulses(audio, result["cancelled"], sr=sr,
                                                        profile=profile)
    else:
        # stream mode: synthesise both waveforms from event lists
        original_audio  = render_to_audio(stream, sr=sr, seed=99)
        processed_audio = render_to_audio(result, sr=sr, seed=99)

    sf.write(os.path.join(out_dir, "original.wav"),  original_audio,  sr)
    sf.write(os.path.join(out_dir, "processed.wav"), processed_audio, sr)

    cancelled_onsets = {p["onset"] for p in result["cancelled"]}
    pulse_data = {
        "duration": stream["duration"],
        "pulses": [
            {"onset": p["onset"], "duration": p["duration"],
             "cancelled": p["onset"] in cancelled_onsets}
            for p in stream["pulses"]
        ],
    }
    with open(os.path.join(out_dir, "pulses.json"), "w") as f:
        json.dump(pulse_data, f)

    _plot_waveform(original_audio, det.event_log,
                   os.path.join(out_dir, "waveform.png"),
                   sr=sr, title=title, cancelled=result["cancelled"])

    summary = _summary(det.event_log, stream, result)
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(summary + "\n")

    return det.event_log, result


# ------------------------------------------------------ scenario runner

def run_scenario(n, mode="stream", patience=3, tolerance=0.25, strength=1.0):
    gen_stream, gt = generate_scenario(n)

    if mode == "audio":
        # render to WAV then extract — exercises the extractor path
        audio   = render_to_audio(gen_stream, sr=SR, seed=99)
        stream  = extract_events(audio, sr=SR)
        orig    = gen_stream   # kept for exact ANC subtraction
        suffix  = "_audio"
    else:
        audio   = None
        stream  = gen_stream
        orig    = None
        suffix  = ""

    label   = SCENARIO_NAMES.get(n, f"Scenario {n}")
    out_dir = os.path.join(os.path.dirname(__file__), "docs", f"scenario_{n}{suffix}")
    title   = f"Scenario {n} [{mode}]: {label}"

    event_log, result = _run_and_save(stream, audio, SR, out_dir, title,
                                      patience, tolerance, strength,
                                      original_stream=orig)
    summary = _summary(event_log, stream, result)
    print(f"Scenario {n} [{mode}] ({label}): {summary}")
    print(f"  → {out_dir}/")
    return event_log, gt


# --------------------------------------------------- real WAV file runner

def run_audio_file(path, patience=3, tolerance=0.25, strength=1.0):
    audio, file_sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    stream  = extract_events(audio, sr=file_sr)
    name    = os.path.splitext(os.path.basename(path))[0]
    out_dir = os.path.join(os.path.dirname(__file__), "docs", f"audio_{name}")

    # Detected durations are noisier in high-background recordings because the threshold
    # captures different fractions of each snore's rise/fall.  Widen tolerance so the
    # detector doesn't break on duration variance that is just noise artefact.
    if stream.get("background_amp", 0) >= 0.005:
        tolerance = max(tolerance, 0.6)

    display_title = _AUDIO_DISPLAY_NAMES.get(name, name)
    event_log, result = _run_and_save(stream, audio, file_sr, out_dir,
                                      display_title, patience, tolerance, strength)
    summary = _summary(event_log, stream, result)
    print(f"Audio '{name}': {len(stream['pulses'])} extracted, "
          f"{len(result['cancelled'])} cancelled")
    print(f"  {summary}")
    print(f"  → {out_dir}/")
    return event_log


# ---------------------------------------------------- dashboard generator

_AUDIO_DISPLAY_NAMES = {
    "real-friend-snoring-extended": "Real Friend Snoring (Quiet Room)",
}


def make_dashboard(outputs_root=None):
    """Generate docs/index.html — waveforms, audio players, remarks per scenario."""
    if outputs_root is None:
        outputs_root = os.path.join(os.path.dirname(__file__), "docs")

    pulse_js_data = {}   # strip_id → {duration, pulses}

    def _load_dir(d, label, name):
        summary_path = os.path.join(outputs_root, d, "summary.txt")
        summary = open(summary_path).read().strip() if os.path.exists(summary_path) else ""
        return {"dir": d, "summary": summary, "name": name, "label": label}

    def _audio_row(label, src, strip_id):
        return f"""
          <div class="audio-row">
            <span>{label}</span>
            <div class="audio-wrap">
              <div class="pulse-strip" id="{strip_id}"><div class="pulse-playhead"></div></div>
              <audio class="audio-with-strip" data-strip="{strip_id}" controls preload="metadata" src="{src}"></audio>
            </div>
          </div>"""

    def _card(card_id, title, modes):
        cols = ""
        for info in modes:
            d = info["dir"]
            orig_strip = f"strip-{d}-orig"
            proc_strip = f"strip-{d}-proc"
            pulses_path = os.path.join(outputs_root, d, "pulses.json")
            if os.path.exists(pulses_path):
                with open(pulses_path) as fj:
                    pdata = json.load(fj)
                pulse_js_data[orig_strip] = pdata
                pulse_js_data[proc_strip] = pdata
            cols += f"""
            <div class="mode-col">
              {"" if not info['label'] else f'<div class="mode-label">{info["label"]}</div>'}
              <div class="summary-text">{info['summary']}</div>
              <img src="{d}/waveform.png" class="waveform" loading="lazy">
              {_audio_row("Original",  f"{d}/original.wav",  orig_strip)}
              {_audio_row("Processed", f"{d}/processed.wav", proc_strip)}
            </div>"""
        return f"""
        <div class="card" id="{card_id}">
          <h2>{title}</h2>
          <div class="modes">{cols}</div>
          <div class="remarks-area">
            <label>Observations for Claude</label>
            <p class="remarks-hint">Type here, hit "Save remarks.txt", then share the file with Claude — it reads your notes and iterates on the results.</p>
            <textarea class="remarks" data-id="{card_id}" rows="3"
              placeholder="e.g. 'The second snore is barely audible after processing but the third is still clearly there...'"></textarea>
          </div>
        </div>"""

    cards_html = ""

    # real audio files first (audio_* directories, sorted by name)
    real_dirs = sorted(
        d for d in os.listdir(outputs_root)
        if d.startswith("audio_") and os.path.isdir(os.path.join(outputs_root, d))
    )
    for d in real_dirs:
        name = d[len("audio_"):]
        display = _AUDIO_DISPLAY_NAMES.get(name, name)
        info = _load_dir(d, "", name)
        cards_html += _card(d, display, [info])

    # synthetic scenarios — collapsed by default so real audio dominates the demo view
    scenario_html = ""
    for n in range(1, 9):
        d = f"scenario_{n}_audio"
        if os.path.isdir(os.path.join(outputs_root, d)):
            modes = [_load_dir(d, "", SCENARIO_NAMES.get(n, f"Scenario {n}"))]
            scenario_html += _card(f"s{n}", f"Scenario {n} — {modes[0]['name']}", modes)
    if scenario_html:
        cards_html += f"""
        <details class="scenarios-group">
          <summary>Test Scenarios</summary>
          {scenario_html}
        </details>"""

    pulse_json = json.dumps(pulse_js_data)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Snore Suppressor</title>
<style>
  :root {{
    --bg: #f5f5f5; --card: #fff; --border: #ddd;
    --green: #4caf50; --red: #e53935; --orange: #ff6600;
    --text: #222; --muted: #666;
  }}
  body {{ margin: 0; padding: 16px; background: var(--bg); font-family: system-ui, sans-serif;
          color: var(--text); font-size: 14px; }}
  .page-header {{ margin-bottom: 12px; }}
  h1   {{ margin: 0 0 4px; font-size: 22px; }}
  .page-subtitle {{ margin: 0 0 12px; color: var(--muted); font-size: 13px; }}
  .toolbar {{ display: flex; gap: 8px; margin-bottom: 16px; align-items: center; flex-wrap: wrap; }}
  .toolbar button {{ padding: 6px 14px; border: none; border-radius: 4px;
                     background: var(--green); color: #fff; cursor: pointer; font-size: 13px; }}
  .toolbar button:hover {{ opacity: 0.85; }}
  .toolbar .hint {{ color: var(--muted); font-size: 12px; }}
  .card  {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px;
            padding: 16px; margin-bottom: 16px; }}
  h2   {{ margin: 0 0 12px; font-size: 16px; }}
  .modes {{ display: flex; gap: 16px; flex-wrap: wrap; }}
  .mode-col {{ flex: 1; min-width: 320px; }}
  .mode-label {{ font-size: 11px; font-weight: 700; text-transform: uppercase;
                 color: var(--muted); margin-bottom: 4px; letter-spacing: 0.5px; }}
  .summary-text {{ font-size: 12px; color: var(--muted); margin-bottom: 8px;
                   background: #f9f9f9; border-radius: 4px; padding: 6px 8px;
                   border-left: 3px solid var(--green); }}
  .waveform {{ width: 100%; border-radius: 4px; margin-bottom: 8px; }}
  .audio-row {{ display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }}
  .audio-row span {{ font-size: 12px; width: 68px; color: var(--muted); flex-shrink: 0; }}
  .audio-wrap {{ flex: 1; display: flex; flex-direction: column; gap: 2px; }}
  audio {{ width: 100%; height: 32px; }}
  .pulse-strip {{
    position: relative; height: 10px; border-radius: 3px;
    background: #e0e0e0; overflow: hidden; cursor: default;
  }}
  .pulse-block {{
    position: absolute; top: 0; bottom: 0; border-radius: 1px;
  }}
  .pulse-block.cancelled {{ background: #ff6600; opacity: 0.85; }}
  .pulse-block.kept      {{ background: #4caf50; opacity: 0.75; }}
  .pulse-playhead {{
    position: absolute; top: 0; bottom: 0; width: 2px; left: 0;
    background: #333; opacity: 0.7; z-index: 5; pointer-events: none;
  }}
  .remarks-area {{ margin-top: 12px; border-top: 1px solid var(--border); padding-top: 10px; }}
  .remarks-area label {{ font-size: 12px; font-weight: 700; color: var(--text);
                          display: block; margin-bottom: 4px; }}
  .remarks-hint {{ margin: 0 0 6px; font-size: 11px; color: var(--muted); font-style: italic; }}
  textarea.remarks {{ width: 100%; box-sizing: border-box; border: 1px solid var(--border);
                      border-radius: 4px; padding: 6px 8px; font-size: 13px;
                      resize: vertical; font-family: inherit; }}
  textarea.remarks:focus {{ outline: none; border-color: var(--green); }}
  .scenarios-group {{ margin-bottom: 16px; }}
  .scenarios-group > summary {{
    cursor: pointer; padding: 10px 14px; border-radius: 6px;
    background: var(--card); border: 1px solid var(--border);
    font-size: 13px; font-weight: 600; color: var(--muted);
    list-style: none; display: flex; align-items: center; gap: 8px;
    user-select: none;
  }}
  .scenarios-group > summary::before {{ content: "▶"; font-size: 10px; transition: transform 0.15s; }}
  .scenarios-group[open] > summary::before {{ transform: rotate(90deg); }}
  .scenarios-group > summary::-webkit-details-marker {{ display: none; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg: #1a1a1a; --card: #242424; --border: #383838;
             --text: #e8e8e8; --muted: #999; }}
    textarea.remarks {{ background: #2a2a2a; color: var(--text); }}
    .summary-text {{ background: #2a2a2a; }}
    .pulse-strip {{ background: #333; }}
    .pulse-playhead {{ background: #fff; opacity: 0.8; }}
  }}
</style>
</head>
<body>
<div class="page-header">
  <h1>🔇 Snore Suppressor</h1>
  <p class="page-subtitle">A PoC — detect rhythmic snoring, learn its signature, cancel it.</p>
</div>
<div class="toolbar">
  <button onclick="exportRemarks()">Save remarks.txt</button>
  <span class="hint">
    <strong>Save remarks.txt → share with Claude → it reads your notes and iterates on the results.</strong>
    <span style="margin-left:16px">
      <span style="display:inline-block;width:10px;height:10px;background:#ff6600;border-radius:2px;vertical-align:middle"></span> cancelled &nbsp;
      <span style="display:inline-block;width:10px;height:10px;background:#4caf50;border-radius:2px;vertical-align:middle"></span> kept
    </span>
  </span>
</div>

{cards_html}

<script>
var PULSE_DATA = {pulse_json};

document.addEventListener('DOMContentLoaded', function() {{
  // Build pulse blocks for each strip
  Object.keys(PULSE_DATA).forEach(function(stripId) {{
    var strip = document.getElementById(stripId);
    var data  = PULSE_DATA[stripId];
    if (!strip || !data || !data.pulses) return;
    var head = strip.querySelector('.pulse-playhead');
    data.pulses.forEach(function(p) {{
      var block = document.createElement('div');
      block.className = 'pulse-block ' + (p.cancelled ? 'cancelled' : 'kept');
      block.style.left  = (p.onset    / data.duration * 100) + '%';
      block.style.width = Math.max(p.duration / data.duration * 100, 0.4) + '%';
      strip.insertBefore(block, head);
    }});
  }});

  // Playhead sync
  document.querySelectorAll('audio.audio-with-strip').forEach(function(audio) {{
    var strip = document.getElementById(audio.dataset.strip);
    if (!strip) return;
    var head = strip.querySelector('.pulse-playhead');
    audio.addEventListener('timeupdate', function() {{
      if (!audio.duration) return;
      head.style.left = (audio.currentTime / audio.duration * 100) + '%';
    }});
    // click strip to seek
    strip.addEventListener('click', function(e) {{
      if (!audio.duration) return;
      var rect = strip.getBoundingClientRect();
      audio.currentTime = ((e.clientX - rect.left) / rect.width) * audio.duration;
    }});
    strip.style.cursor = 'pointer';
  }});

  // Remarks persistence
  document.querySelectorAll('.remarks').forEach(function(ta) {{
    var key = 'snore:remarks:' + ta.dataset.id;
    ta.value = localStorage.getItem(key) || '';
    ta.addEventListener('input', function() {{ localStorage.setItem(key, ta.value); }});
  }});
}});

function exportRemarks() {{
  var lines = [];
  document.querySelectorAll('.remarks').forEach(function(ta) {{
    var text = ta.value.trim();
    if (text) {{
      lines.push('[' + ta.dataset.id + ']');
      lines.push(text);
      lines.push('');
    }}
  }});
  if (!lines.length) {{ alert('No remarks yet.'); return; }}
  var blob = new Blob([lines.join('\\n')], {{type: 'text/plain'}});
  var a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'remarks.txt';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}}
</script>
</body>
</html>"""

    out_path = os.path.join(outputs_root, "index.html")
    with open(out_path, "w") as f:
        f.write(html)
    print(f"Dashboard written → {out_path}")
    return out_path


# ---------------------------------------------------- sample generator

def make_samples(out_dir="samples"):
    os.makedirs(out_dir, exist_ok=True)
    for n in range(1, 9):
        stream, _ = generate_scenario(n)
        audio     = render_to_audio(stream, sr=SR)
        path      = os.path.join(out_dir, f"scenario_{n}.wav")
        sf.write(path, audio, SR)
        print(f"  Wrote {path}  ({len(stream['pulses'])} pulses, {stream['duration']:.0f}s)")


# ----------------------------------------------------------------------- CLI

def main():
    parser = argparse.ArgumentParser(description="Snore pattern-detection test harness")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--scenario",     type=int,  metavar="N", help="Run scenario N (1–8)")
    group.add_argument("--all",          action="store_true",    help="Run all 8 scenarios")
    group.add_argument("--audio",        metavar="FILE",         help="Process a WAV file directly")
    group.add_argument("--make-samples", action="store_true",    help="Write samples/ WAVs from all scenarios")
    group.add_argument("--dashboard",    action="store_true",    help="Regenerate docs/index.html")

    parser.add_argument("--mode",      choices=["stream", "audio"], default="stream",
                        help="stream: exact event list (default);  audio: render→extract→detect")
    parser.add_argument("--patience",  type=int,   default=3)
    parser.add_argument("--tolerance", type=float, default=0.25)
    parser.add_argument("--strength",  type=float, default=1.0)
    parser.add_argument("--check",     action="store_true",
                        help="Run automated pass/fail checks after --scenario / --all")
    args = parser.parse_args()

    if args.make_samples:
        print("Generating sample WAV files...")
        make_samples()
        return

    if args.dashboard:
        make_dashboard()
        return

    if args.audio:
        run_audio_file(args.audio, args.patience, args.tolerance, args.strength)
        return

    scenarios = range(1, 9) if args.all else [args.scenario]
    for n in scenarios:
        run_scenario(n, args.mode, args.patience, args.tolerance, args.strength)

    if args.check or args.all:
        print("\nRunning automated checks...")
        ok = run_all_checks(args.patience, args.tolerance, args.strength)
        if not ok:
            sys.exit(1)

    if args.all:
        make_dashboard()


if __name__ == "__main__":
    main()
