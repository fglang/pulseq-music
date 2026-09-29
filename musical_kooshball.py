"""
Musical kooshball: audio-based 3D radial-out MRI sequences in Pulseq.

Convert audio waveforms into a koosh ball trajectory while aiming to preserve
the sound of the original music.

steps:
1. Apply low-pass filter to audio.
2. Apply multiplicative envelope to create zero-gradient gaps for RF pulses.
3. Suppress forbidden frequency bands by convex optimization while preserving
   the RF gaps.
4. Add variable half-sine compensation gradients to keep a constant
   gradient moment spoke.
5. Distribute spoke endpoints over the sphere along a smoothly varying
   spherical spiral with approximately constant angular increments.

SAFETY NOTICE

This software generates experimental MRI pulse sequences for research use only.

Before running a generated sequence on an MRI system, independently verify
compliance with all applicable hardware and safety limits. The user is
responsible for this verification. In particular, check gradient waveforms for
excitation of scanner-specific mechanical and acoustic resonances, which may
cause excessive vibration, acoustic noise, or damage to the gradient system.

Do not assume that the resonance bands or safety limits in this example apply
to another scanner. Always use the limits specified for the individual MRI
system and exercise caution when testing sequences on a scanner.

"""

import argparse
import math
import subprocess
from pathlib import Path

import numpy as np
import pypulseq as pp
import matplotlib.pyplot as plt
from pypulseq.utils.siemens.asc_to_hw import asc_to_acoustic_resonances
from pypulseq.utils.siemens.readasc import readasc
from scipy.fft import irfft, rfft, rfftfreq
from scipy.io import wavfile
from scipy.signal import butter, resample_poly, sosfiltfilt
from scipy.sparse.linalg import LinearOperator, cg

# these forbidden frequency bands are used if no ASC file is provided
# make sure to adapt them for your specific system!
DEFAULT_BANDS = [{"frequency": 576, "bandwidth": 100}, {"frequency": 1090, "bandwidth": 300}]

# scanner limits
system = pp.Opts(max_grad=30, grad_unit="mT/m", max_slew=140, slew_unit="T/m/s",
                    rf_ringdown_time=30e-6, rf_dead_time=100e-6,
                    adc_dead_time=10e-6, grad_raster_time=10e-6)

def radial3d_directions(n, full_sphere=True):
    """Direction vectors for smoothly varying spherical spiral
    """
    nn = np.arange(n)
    z = 1 - (2 if full_sphere else 1) * (nn + 0.5) / n
    area_per_point = (4 if full_sphere else 2) * np.pi / n
    rxy = np.sqrt(np.maximum(0, 1 - z**2))
    cos_dphi = (np.cos(np.sqrt(area_per_point)) - z[:-1] * z[1:]) / (rxy[:-1] * rxy[1:])
    phi = np.r_[0, np.cumsum(np.arccos(np.clip(cos_dphi, -1, 1)))]
    order = nn
    
    return np.array([rxy * np.cos(phi), rxy * np.sin(phi), z])[:, order]


def half_sine(area, n, dt):
    s = np.sin(np.pi * (np.arange(n) + 0.5) / n)
    return s * (area / (s.sum() * dt))


def load_audio(path):
    """WAV via SciPy, other formats via ffprobe/ffmpeg"""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.load(path), None  # already-resampled yf from earlier processing
    if path.suffix.lower() == ".wav":
        fs, y = wavfile.read(path)
        if y.dtype.kind in "iu":
            if y.dtype.kind == "u":
                y = (y.astype(np.float64) - 2 ** (8 * y.dtype.itemsize - 1)) / 2 ** (8 * y.dtype.itemsize - 1)
            else:
                y = y.astype(np.float64) / 2 ** (8 * y.dtype.itemsize - 1)
        return y, fs
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=sample_rate,channels", "-of", "csv=p=0", str(path)],
        check=True, capture_output=True, text=True,
    )
    fs, channels = map(int, probe.stdout.strip().split(","))
    decoded = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0", "-f", "f32le", "-acodec", "pcm_f32le", "-"],
        check=True, capture_output=True,
    )
    return np.frombuffer(decoded.stdout, dtype="<f4").reshape(-1, channels), fs


def prepare_audio(path, tmin, tmax, lowpass_hz, dt):
    """crop to time range, low pass filter, stereo->mono"""
    y, fs = load_audio(path)
    if fs is None:
        return np.asarray(y, dtype=float).mean(axis=1) if y.ndim == 2 else np.asarray(y, dtype=float)
    # crop to desired time interval
    first = int(np.floor(tmin * fs + 0.5))
    stop = int(np.floor(tmax * fs + 0.5)) - 1
    if stop <= first or stop > len(y):
        raise ValueError("audio does not cover the selected tmin:tmax interval")
    y = y[first:stop]
    up = int(np.floor((1 / dt / fs) * 100 + 0.5))
    y = resample_poly(y, up, 100, axis=0)
    if lowpass_hz > 0:
        filter_fs = 1 / dt
        sos = butter(8, lowpass_hz, fs=filter_fs, output="sos")
        y = sosfiltfilt(sos, y, axis=0)
    return y.mean(axis=1) if y.ndim == 2 else y


def read_stopbands(asc_path):
    """Siemens-specific"""
    if asc_path is None:
        return DEFAULT_BANDS
    asc, _ = readasc(str(asc_path))
    bands = asc_to_acoustic_resonances(asc)
    if not bands:
        raise ValueError(f"No acoustic resonance bands in {asc_path}")
    return bands


def optimize_audio(y, n_tr, n_active, hull, bands, fs, lam, rtol=1e-6, maxiter=100):
    """
    suppress forbidden frequencies while keeping RF gaps with zero gradients
    -> convex optimization problem solved by conjugate gradients
    see README for details
    """
    n_input = len(y)
    n_blocks = (n_input + n_active - 1) // n_active
    n_out = n_blocks * n_tr
    freq = rfftfreq(n_out, 1 / fs)
    mask = np.zeros(len(freq), dtype=bool)
    for band in bands:
        center, width = band["frequency"], band["bandwidth"]
        mask |= (freq >= center - width / 2) & (freq <= center + width / 2)

    def forward(x): # gap insertion & envelope ("hull")
        padded = np.zeros(n_blocks * n_active)
        padded[:n_input] = x
        blocks = np.zeros((n_blocks, n_tr))
        blocks[:, :n_active] = padded.reshape(n_blocks, n_active) * hull
        return blocks.ravel()

    def normal(x):
        g = forward(x)
        projected = irfft(rfft(g) * mask, n=n_out) # project onto forbidden frequency bands
        return x + lam * (projected.reshape(n_blocks, n_tr)[:, :n_active] * hull).ravel()[:n_input]

    iters = 0

    def step(_):
        nonlocal iters
        iters += 1

    x, flag = cg(LinearOperator((n_input, n_input), matvec=normal, dtype=float), y,
                 x0=y.copy(), rtol=rtol, atol=0, maxiter=maxiter, callback=step)
    g = forward(x)

    # output some diagnostics
    reference = rfft(forward(y))
    result = rfft(g)
    # Weight positive-frequency bins twice, except DC/Nyquist.
    weights = np.full(len(freq), 2.0)
    weights[0] = 1.0
    if n_out % 2 == 0:
        weights[-1] = 1.0
    p_ref = np.sum(weights[mask] * abs(reference[mask]) ** 2)
    p_opt = np.sum(weights[mask] * abs(result[mask]) ** 2)
    print(f"CG iterations {iters}, flag {flag}, relative residual {np.linalg.norm(normal(x) - y) / np.linalg.norm(y):.3g}")
    print(f"Relative audio change: {np.linalg.norm(x - y) / np.linalg.norm(y):.3g}")
    print(f"Forbidden-band power reduction: {10 * np.log10(max(p_opt / p_ref, np.finfo(float).eps)):.1f} dB")
    return g.reshape(n_blocks, n_tr), n_blocks


def make_sequence(args):
    """main assembly of pypulseq sequence"""
    dt = system.grad_raster_time
    rf = pp.make_block_pulse(np.deg2rad(args.FA), delay=system.rf_dead_time,
                             duration=args.rfdur, use="excitation", system=system)
    gap = pp.calc_duration(rf)
    n_tr = int(np.floor(args.tr / dt + 0.5))
    n_gap = int(np.floor(gap / dt + 0.5))
    if abs(gap - n_gap * dt) > 1e-12:
        raise ValueError("RF block duration is not on the gradient raster")
    n_active = n_tr - n_gap
    if n_active < 2:
        raise ValueError("TR leaves no room for gradient waveform")
    hull = abs(np.sin(np.pi * np.arange(n_active) / (n_active - 1)))
    y = prepare_audio(args.audio, args.tmin, args.tmax, args.lowpass, dt)
    bands = read_stopbands(args.asc)
    print("Resonance bands (Hz):", [(b["frequency"], b["bandwidth"]) for b in bands])
    blocks, nspokes = optimize_audio(y, n_tr, n_active, hull, bands, 1 / dt, args.lam)
    directions = radial3d_directions(nspokes, args.full_sphere)

    # scale waveform according to system limits
    gmax = system.max_grad / system.gamma   # T/m
    smax = system.max_slew / system.gamma   # T/m/s
    w = blocks / np.max(np.abs(blocks)) * gmax
    max_slew = np.max(np.abs(np.diff(w.ravel()))) / dt
    w *= min(0.9 * smax / max_slew, 0.9 * gmax / np.max(np.abs(w))) # some safety margin here...
    active_hz = w[:, :n_active] * system.gamma
    print(f"Audio gradient peak: {1e3 * np.max(abs(w[:, :n_active])):.2f} mT/m")
    adc_dwell = args.adcdwell
    adc_delay = system.adc_dead_time
    # ADC dead time at the end must fit before the following RF block.
    nro = math.floor((n_active * dt - adc_delay - system.adc_dead_time + 1e-12) / adc_dwell)
    adc = pp.make_adc(nro, dwell=adc_dwell, delay=adc_delay, system=system)
    if pp.calc_duration(adc) > n_active * dt + 1e-12:
        raise ValueError("ADC overruns gradient block")

    seq = pp.Sequence(system)
    if args.audiomom == "none":
        wanted_area = None
    else:
        wanted_area = 0.0 if args.audiomom == "balanced" else float(args.audiomom)

    stereo = np.zeros((nspokes, n_tr, 2), dtype=np.float32) # gather for wav export later

    for j in range(nspokes):
        gaudio = active_hz[j]
        if wanted_area is not None:
            delta_area = wanted_area - gaudio.sum() * dt
            compensation = half_sine(delta_area, n_active, dt) 
            gaudio = gaudio + compensation
        gx, gy, gz = (pp.make_arbitrary_grad(axis, gaudio * direction, first=0, last=0, system=system)
                      for axis, direction in zip("xyz", directions[:, j]))
        seq.add_block(rf)
        seq.add_block(gx, gy, gz, adc)
        
        # audio: x->L, y->R, z to both.
        stereo[j, n_gap:, 0] = (gx.waveform + gz.waveform) / system.gamma
        stereo[j, n_gap:, 1] = (gy.waveform + gz.waveform) / system.gamma

    seq.set_definition("nspokes", nspokes)
    seq.set_definition("audiomom", args.audiomom if args.audiomom in ("balanced", "none") else wanted_area)
    seq.set_definition("TR", n_tr * dt)
    seq.set_definition("rfdur", args.rfdur)
    seq.set_definition("FA", args.FA)
    seq.set_definition("adcdwell", args.adcdwell)
    seq.set_definition("audiofile", str(args.audio))
    ok, errors = seq.check_timing()
    if not ok:
        raise RuntimeError("Pulseq timing failed:\n" + "\n".join(map(str, errors[:10])))
    print(f"RF block {1e3 * gap:.3f} ms; gradient block {1e3 * n_active * dt:.3f} ms; TR {1e3 * n_tr * dt:.3f} ms")
    seq.write(str(args.output))
    print(f"Wrote {args.output}, {nspokes} spokes, {nro} ADC samples/spoke")

    # write simulated sound to wav file
    rendered = resample_poly(stereo.reshape(-1, 2), 441, 1000, axis=0)
    peak = np.max(abs(rendered))
    if peak:
        rendered /= peak
    wavfile.write(str(args.output) + ".wav", 44100, rendered.astype(np.float32))

    seq.calculate_gradient_spectrum(acoustic_resonances=bands, plot=True,
                                    max_frequency=max(2000, max(b["frequency"] + b["bandwidth"] / 2 for b in bands) + 200))
    plt.savefig(str(args.output) + "_spectrum.png", dpi=160)
    return seq


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, default=Path("audio/mozart_nachtmusik.ogg"), help="input audio file")
    parser.add_argument("--asc", type=Path, default=None, help="Siemens ASC file. if omitted: two bands hard-coded in the beginning of the script (*be careful with your system!*)")
    parser.add_argument("--output", type=Path, default=Path("out/musical_kooshball.seq"))
    parser.add_argument("--tmin", type=float, default=0)
    parser.add_argument("--tmax", type=float, default=49.5)
    parser.add_argument("--tr", type=float, default=6.96e-3, help="RF-to-RF interval, seconds.") # TR=6.96ms ~corresponds to a musically fitting note d pedal point (5th in the key of g major ;)
    parser.add_argument("--adcdwell", type=float, default=5e-6, help="ADC dwell time")
    parser.add_argument("--FA", type=float, default=5., help="flip angle (deg)")
    parser.add_argument("--rfdur", type=float, default=20e-6, help="RF pulse duration (s)")
    parser.add_argument("--lam", type=float, default=40, help="regularization parameter to supress forbidden frequencies")
    parser.add_argument("--lowpass", type=float, default=4000)
    parser.add_argument("--audiomom", default=600, help="target gradient moment (1/m), 'balanced', or 'none'")
    parser.add_argument("--hemisphere", dest="full_sphere", action="store_false")
    make_sequence(parser.parse_args())


if __name__ == "__main__":
    main()
