"""Animate a Pulseq sequence's ADC trajectory and gradients with its WAV audio.

Requires numpy, matplotlib, pypulseq, scipy, and ffmpeg.
Example:
    python make_animation.py --sequence musical_kooshball.seq --wav musical_kooshball.seq.wav
    python make_animation.py --sequence musical_kooshball.seq --wav musical_kooshball.seq.wav \
        --start 2 --end 5 --fps 30 --output preview.mp4

uses ffmpeg to write mp4 video with wav audio track included

"""

import argparse
import math
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pypulseq as pp
from scipy.io import wavfile


def gradient_envelopes(seq, start, end, n_bins=1600):
    """Keep gradient extrema visible when many raster samples occupy a pixel.
    Inspired by SeqEyes plugin"""
    waves = seq.waveforms_and_times(time_range=[start, end])[0]
    edges = np.linspace(start, end, n_bins + 1)
    low = np.zeros((3, n_bins))
    high = np.zeros_like(low)
    for axis, wave in enumerate(waves[:3]):
        if wave.size == 0:
            continue
        times, values = wave
        inside = (times >= start) & (times <= end)
        edge_values = np.interp(edges, times, values, left=0, right=0)
        times = np.r_[times[inside], edges[:-1], edges[1:]]
        values = np.r_[values[inside], edge_values[:-1], edge_values[1:]]
        bins = np.clip(np.searchsorted(edges, times, side="right") - 1, 0, n_bins - 1)
        np.minimum.at(low[axis], bins, values)
        np.maximum.at(high[axis], bins, values)
    return edges, low / seq.system.gamma * 1e3, high / seq.system.gamma * 1e3


def trajectory(seq, start, end, max_points, points_per_readout):
    k, _, _, _, times = seq.calculate_kspace() # time consuming for long sequences...
    times = np.asarray(times).ravel()
    if k.shape[0] != 3 or k.shape[1] != len(times):
        raise ValueError("Expected a 3D trajectory with one coordinate per ADC sample")
    selected = np.flatnonzero((times >= start) & (times < end))
    if len(selected) == 0:
        raise ValueError("No ADC samples in the selected time range")

    # Explicit ADC event boundaries avoid drawing a line across successive spokes.
    counts = []
    for block_id in seq.block_events:
        adc = seq.get_block(block_id).adc
        if adc is not None:
            counts.append(adc.num_samples)
    if sum(counts) != len(times):
        raise ValueError("ADC event counts do not match calculated k-space samples")

    indices = selected[np.linspace(0, len(selected) - 1, min(len(selected), max_points), dtype=int)]
    cloud_k, cloud_t = k[:, indices], times[indices]
    path_k, path_t = [], []
    first = 0
    for count in counts:
        last = first + count
        if times[last - 1] >= start and times[first] < end:
            indices = np.arange(first, last)
            indices = indices[(times[indices] >= start) & (times[indices] < end)]
            if len(indices) == 0:
                first = last
                continue
            indices = indices[np.linspace(0, len(indices) - 1,
                                          min(len(indices), points_per_readout), dtype=int)]
            path_k.append(k[:, indices])
            path_k.append(np.full((3, 1), np.nan))
            path_t.extend((times[indices], [times[indices[-1]]]))
        first = last
    return cloud_k, cloud_t, np.hstack(path_k), np.concatenate(path_t), float(np.max(abs(k[:, selected])))


def render(args):
    seq = pp.Sequence()
    seq.read(str(args.sequence))
    duration = seq.duration()[0]
    start = args.start
    end = min(duration, args.end if args.end is not None else duration)
    if not 0 <= start < end:
        raise ValueError(f"Time range must overlap the sequence (duration {duration:g} s)")
    sample_rate, wav = wavfile.read(args.wav, mmap=True)
    if len(wav) / sample_rate < end - 1 / sample_rate:
        raise ValueError("WAV is shorter than the selected sequence interval")
    del wav

    print("Calculating ADC trajectory ...", flush=True)
    cloud_k, cloud_t, path_k, path_t, peak_k = trajectory(
        seq, start, end, args.max_points, args.points_per_readout)
    print("Preparing gradient overview ...", flush=True)
    edges, low, high = gradient_envelopes(seq, start, end)
    n_frames = math.ceil((end - start) * args.fps)

    # define colors here
    bg, fg, cyan = "#060910", "#d2deef", "#33defe"
    colors = ("#63c25c", "#638cf0", "#f0a638")

    fig = plt.figure(figsize=(args.width / 100, args.height / 100), dpi=100, facecolor=bg)
    ax = fig.add_axes([0.09, 0.43, 0.82, 0.52], projection="3d", facecolor=bg)
    limit = max(1, 1.06 * peak_k)
    for set_limit in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
        set_limit(-limit, limit)
    ax.set_box_aspect((1, 1, 1))
    for label, setter in zip(("kₓ (1/m)", "kᵧ (1/m)", "kᶻ (1/m)"),
                             (ax.set_xlabel, ax.set_ylabel, ax.set_zlabel)):
        setter(label, color=fg)
    ax.tick_params(colors=fg, labelsize=8)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.line.set_color("#526071")
    ax.grid(alpha=0.15)
    ax.view_init(elev=23, azim=-35)
    cloud, = ax.plot([], [], [], ".", color="#304b62", markersize=1.2)
    trails = [ax.plot([], [], [], color=cyan, alpha=0.25 + 0.2 * j,
                      linewidth=0.7 + j * 0.25)[0] for j in range(4)]
    dot, = ax.plot([], [], [], "o", color="white", markersize=4)
    title = fig.text(0.5, 0.965, "", color=fg, fontsize=13, ha="center", va="top")

    rows = []
    scale = max(1, np.max(np.abs([low, high]))) * 1.1
    for axis in range(3):
        row = fig.add_axes([0.12, 0.075 + (2 - axis) * 0.105, 0.81, 0.085], facecolor=bg)
        row.fill_between(edges[:-1], low[axis], high[axis], step="post", color=colors[axis], alpha=0.35)
        row.step(edges[:-1], low[axis], where="post", color=colors[axis], lw=0.5)
        row.step(edges[:-1], high[axis], where="post", color=colors[axis], lw=0.5)
        row.axhline(0, color="#526071", lw=0.5)
        cursor = row.axvline(start, color="white", lw=1.3)
        row.set_xlim(start, end)
        row.set_ylim(-scale, scale)
        row.set_ylabel("G" + "xyz"[axis] + "\n(mT/m)", rotation=0, labelpad=28,
                       color=colors[axis], va="center", fontsize=9)
        row.set_yticks([-round(scale, 1), 0, round(scale, 1)])
        row.tick_params(colors=fg, labelsize=8, length=0)
        row.grid(axis="x", color="#314050", alpha=0.5)
        for spine in row.spines.values():
            spine.set_visible(False)
        if axis < 2:
            row.tick_params(labelbottom=False)
        else:
            row.set_xlabel("sequence time (s)", color=fg)
        rows.append(cursor)

    def set_xyz(line, xyz):
        line.set_data(xyz[:2])
        line.set_3d_properties(xyz[2])

    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{args.width}x{args.height}",
               "-r", str(args.fps), "-i", "-", "-ss", str(start), "-i", str(args.wav),
               "-map", "0:v:0", "-map", "1:a:0", "-vf", "format=yuv420p",
               "-c:v", "libx264", "-crf", "18", "-preset", "medium",
               "-c:a", "aac", "-b:a", "192k", "-af", "apad",
               "-t", str(n_frames / args.fps), "-movflags", "+faststart", str(args.output)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for frame in range(n_frames):
            t = min(start + (frame + 1) / args.fps, end)
            n = np.searchsorted(cloud_t, t, side="right")
            set_xyz(cloud, cloud_k[:, :n])
            bounds = np.linspace(max(start, t - args.trail), t, 5)
            for j, line in enumerate(trails):
                lo, hi = np.searchsorted(path_t, bounds[j:j + 2], side="right")
                set_xyz(line, path_k[:, lo:hi])
            i = np.searchsorted(path_t, t, side="right") - 1
            while i >= 0 and np.isnan(path_k[0, i]):
                i -= 1
            set_xyz(dot, path_k[:, i:i + 1] if i >= 0 and t - path_t[i] < 1 / args.fps
                    else np.empty((3, 0)))
            for cursor in rows:
                cursor.set_xdata([t, t])
            title.set_text(f"The sound of k-space    {t - start:.2f} / {end - start:.2f} s")
            ax.view_init(elev=23, azim=-35 + args.rotate * (t - start) / (end - start))
            fig.canvas.draw()
            process.stdin.write(fig.canvas.buffer_rgba())
            if (frame + 1) % args.fps == 0 or frame + 1 == n_frames:
                print(f"  {frame + 1} / {n_frames} frames", flush=True)
        process.stdin.close()
        error = process.stderr.read().decode(errors="replace")
        if process.wait() != 0:
            raise RuntimeError("FFmpeg failed:\n" + error)
    except Exception:
        process.kill()
        process.wait()
        raise
    finally:
        plt.close(fig)
    print(f"Created {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sequence", type=Path, default=Path('out/musical_kooshball.seq'), help="Pulseq .seq file")
    parser.add_argument("--wav", type=Path, default=Path('out/musical_kooshball.seq.wav'), help="WAV synthesized from the same full sequence")
    parser.add_argument("--output", type=Path, default=Path("out/musical_kooshball.mp4"))
    parser.add_argument("--start", type=float, default=0, help="Start on the sequence timeline (s)")
    parser.add_argument("--end", type=float, help="End on the sequence timeline (s)")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--trail", type=float, default=0.12, help="highlighted recent trajectory duration (s)")
    parser.add_argument("--rotate", type=float, default=0, help="Camera rotation across clip (degrees)")
    parser.add_argument("--max-points", type=int, default=10000, help="Maximum background ADC points")
    parser.add_argument("--points-per-readout", type=int, default=100)
    args = parser.parse_args()
    if args.fps < 1 or args.width < 200 or args.height < 200 or args.width % 2 or args.height % 2:
        parser.error("fps must be positive; width and height must be even and at least 200")
    if args.trail <= 0 or args.max_points < 1 or args.points_per_readout < 2:
        parser.error("trail and point limits must be positive (at least 2 per readout)")
    render(args)


if __name__ == "__main__":
    main()
