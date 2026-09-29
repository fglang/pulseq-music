# Musical Kooshball

**Turn music into a 3D radial MRI pulse sequence.** This code translates an audio waveform to the scanner's gradient axes, forming a 3D koosh-ball k-space trajectory. The generated Pulseq sequence can be animated including audio playback.

## What it does

1. Resample and filter the input audio. Arrange it into gradient blocks with gaps for RF excitation.
2. Suppresse selected acoustic resonance bands in the shaped waveform by convex optimization.
3. Add compensation waveforms (half-sine, rather quiet) to achieve a defined gradient moment at the end of each spoke, and rotate spokes in 3D directions.
4. Write a Pulseq `.seq` file, a WAV audio file rendering of its gradients, and a plot of the final gradient spectrum. A separate script animates the ADC trajectory and gradients with the WAV audio.

## Try it

Install the Python dependencies with:

```bash
python -m pip install -r requirements.txt
```

Install [ffmpeg](https://ffmpeg.org/) separately for non-wav audio input (e.g. ogg) and MP4 export. Both `ffmpeg` and `ffprobe` should be available on `PATH`.

Forbidden frequency bands can be read from an .asc file (Siemens-specific) or manually entered at the top of the script.

To generate the sequence:

```bash
python musical_kooshball.py --audio audio/mozart_nachtmusik.ogg --asc path/to/scanner.asc
```

This writes `musical_kooshball.seq`, `musical_kooshball.seq.wav`, and `musical_kooshball.seq_spectrum.png`. For a short animation from the generated sequence:

```bash
python make_animation.py --sequence musical_kooshball.seq --wav musical_kooshball.seq.wav \
    --start 0 --end 10 --output musical_kooshball.mp4
```

If `--asc` is omitted, the example uses two bands defined in `musical_kooshball.py`. **These bands must be adapted to the specific scanner**.

## Safety notice

This repository contains experimental software for generating MRI pulse sequences. It is intended for research and demonstration purposes only and has not been validated for clinical use.

**Any sequence generated with this software must be independently checked before execution on an MRI scanner.** The user is responsible for ensuring that the final sequence is compatible with the hardware and safety limits of the specific MRI system.

Particular care must be taken with **mechanical and acoustic resonances of the gradient system**. Gradient waveforms can contain substantial spectral energy near scanner-specific resonance frequencies. Excitation of these resonances may lead to excessive vibration, acoustic noise, and potential damage to the gradient system.

The code suppresses selected frequency bands, but this does **not** constitute a safety guarantee. Resonance frequencies and prohibited bands are scanner-specific and must be determined from the individual scanner's system information or measurements. Inspect the final gradient spectrum and assess the waveform before execution. When testing a new waveform, use conservative initial gradient amplitudes and appropriate monitoring.

## Forbidden-band suppression

The audio waveform is optimized to suppress forbidden frequency bands *before* it is converted into gradient blocks. Let $y$ be the original audio and $Sx$ the waveform produced from optimized audio $x$ by applying the periodic envelope and inserting zero-gradient RF gaps. The optimization objective is

$$
\min_x \ \frac12\|x-y\|^2+\frac{\lambda}{2}\|M_BFSx\|^2,
$$

where $F$ is the Fourier transform and $M_B$ selects the forbidden frequency bands. Thus, the optimization preserves the audio while reducing forbidden-band power after the envelope and gaps have been applied. Setting the gradient to zero gives

$$
(I+\lambda S^\mathsf{T}P_BS)x=y,
$$

with $P_B=F^\mathsf{H}M_BF$. This is solved by matrix-free conjugate gradients. The final Pulseq gradient spectrum is plotted separately after scaling, moment compensation, and direction assignment.

## Links and inspirations

- **Musical MR fingerprinting sequence:** Ma D et al. *Music-based magnetic resonance fingerprinting to improve patient comfort during MRI examinations.* Magnetic Resonance in Medicine. 2016;75(6):2303-2314 https://doi.org/10.1002/mrm.25818

- **Pulseq**
  - https://pulseq.github.io/
    - **mrMusic**: play melodies on the scanner by defining simple triangular waveforms note-by-note:
    https://github.com/pulseq/pulseq/tree/master/matlab/demoUnsorted/%2BmrMusic
  - https://github.com/pulseq/pypulseq
  - Layton KJ et al. *Pulseq: A rapid and hardware-independent pulse sequence prototyping framework.* Magnetic Resonance in Medicine. 2017;77(4):1544-1552. https://doi.org/10.1002/mrm.26235


## Audio attribution

The example audio is an excerpt from **Wolfgang Amadeus Mozart: *Eine kleine Nachtmusik*, K. 525, I. Allegro**, performed by the **Advent Chamber Orchestra**.

- **Source:** [Wikimedia Commons — *Mozart - Eine kleine Nachtmusik - 1. Allegro.ogg*](https://commons.wikimedia.org/wiki/File:Mozart_-_Eine_kleine_Nachtmusik_-_1._Allegro.ogg)
- **License:** [Creative Commons Attribution-ShareAlike 2.0 (CC BY-SA 2.0)](https://creativecommons.org/licenses/by-sa/2.0/)

The recording is used as input to the Pulseq conversion example. The example video and sequence-derived WAV contain transformed audio from that recording. Credit the source and observe its CC BY-SA terms when redistributing them. The audio license is separate from the [MIT license](LICENSE) covering the source code.
