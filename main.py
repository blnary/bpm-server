import librosa
import math
import time
import numpy as np
from fastapi import FastAPI


class Args:
    def __init__(self, filename, window, thres, decimation, n_fft,
                 win_length, hop_length, use_sine, plot):
        self.filename = filename
        self.window = window
        self.thres = thres
        self.decimation = decimation
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.use_sine = use_sine
        self.plot = plot


app = FastAPI()


@app.get("/bpm/")
async def get_bpm(filename: str):
    args = Args(filename, 4, 0.07, 16, 512, 512, 128, False, False)
    bpm, offset = process_file(args)
    return {"bpm": bpm, "offset": offset}


def gen_sin():
    fs = 44100
    samps = librosa.tone(5678, duration=10, sr=fs)
    return samps, fs


def gcd_spb(spb_candidate, spb_peak):
    spb_candidate = np.sort(spb_candidate)
    if len(spb_candidate) == 0:
        return -1
    min_rel_err = float('inf')
    result = -1

    # Try for every simple fractions
    for i in range(1, 10):
        for j in range(1, 10):
            coef = i / j
            unit = spb_peak * coef

            # Prefer whole BPM
            rel_err = abs(round(60 / unit) - 60 / unit)

            # Limit BPM to 80-240
            if 60 / unit > 240 or 60 / unit < 80:
                continue
            last_spb = -1
            for val in spb_candidate:
                if abs(val - last_spb) < 0.01:
                    continue
                rel_err += 0 if abs(round(val / unit) -
                                    val / unit) < 0.01 else 1
                last_spb = val
            if rel_err < min_rel_err:
                min_rel_err = rel_err
                result = unit
    return result


def detect_bpm(data, fs, args):
    decimation = args.decimation
    min_ndx = math.floor(60.0 / 240 * (fs / decimation))
    max_ndx = math.floor(60.0 / 30 * (fs / decimation))

    # Downsample
    remainder = len(data) % decimation
    zeros_needed = decimation - remainder
    data = np.pad(data, (0, zeros_needed), mode='constant')
    data = np.max(data.reshape(-1, decimation), axis=1)

    # Normalize
    data = data - np.mean(data)

    # ACF
    correl = np.correlate(data, data, "full")
    midpoint = len(correl) // 2
    correl_midpoint_tmp = correl[midpoint + min_ndx:midpoint + max_ndx]

    # Weaken higher tempo
    mult = midpoint - np.arange(min_ndx, max_ndx)
    correl_midpoint_tmp = correl_midpoint_tmp / mult

    # Normalize
    correl_midpoint_tmp = correl_midpoint_tmp / \
        np.linalg.norm(correl_midpoint_tmp)

    # Detect candidate
    high_ndx = np.argwhere(correl_midpoint_tmp > args.thres)
    high_ndx_adjusted = high_ndx + min_ndx
    spb_candidate = high_ndx_adjusted / (fs / decimation)

    # Detect peak
    peak_ndx = np.argmax(correl_midpoint_tmp)
    peak_ndx_adjusted = peak_ndx + min_ndx
    spb_peak = peak_ndx_adjusted / (fs / decimation)
    correl = correl_midpoint_tmp[peak_ndx]

    # Get seconds per beat
    n = np.arange(0, len(correl_midpoint_tmp))
    n = (n + min_ndx) / (fs / decimation)

    # Return
    return spb_candidate, spb_peak, correl


def process_file(args):
    print("Loading file...")
    initial_time = time.time()
    samps, fs = gen_sin() if args.use_sine else librosa.load(args.filename)
    data = []
    spbc = 0
    peak_correl = 0
    peak_spb = 0
    peak_samp_ndx = 0
    spb_candidate = np.array([], dtype=np.float64)
    bpm = 0
    nsamps = len(samps)
    window_samps = int(args.window * fs)
    samps_ndx = 0
    max_window_ndx = math.floor(nsamps / window_samps)

    # Iterate through all windows, collect spb candidates and peak
    print("Doing auto correlation...")
    for window_ndx in range(0, max_window_ndx):
        data = samps[samps_ndx: samps_ndx + window_samps]
        if not ((len(data) % window_samps) == 0):
            raise AssertionError(str(len(data)))
        spbc, spbp, correl = detect_bpm(data, fs, args)
        if spbc is None:
            continue
        spb_candidate = np.append(spb_candidate, spbc)
        if correl > peak_correl:
            peak_correl = correl
            peak_spb = spbp
            peak_samp_ndx = samps_ndx
        samps_ndx = samps_ndx + window_samps

    # Calculate BPM by GCD
    spb = gcd_spb(spb_candidate, peak_spb)
    bpm = 60 / spb
    rounded_bpm = round(bpm)
    rel_err = bpm - rounded_bpm

    # Calculate offset by onset algorithm
    print("Calculating offset...")
    data = samps[peak_samp_ndx: peak_samp_ndx + window_samps]
    onset_env = librosa.onset.onset_strength(
        y=data,
        sr=fs,
        hop_length=args.hop_length,
        n_fft=args.n_fft,
        win_length=args.win_length
    )
    onset_env = np.gradient(onset_env)
    onset_env = onset_env / np.max(onset_env)
    onset_trim = 8
    onset_env = onset_env[onset_trim:]
    raw_offset_ndx = np.argmax(onset_env)
    onset_offset = args.n_fft // (2 * args.hop_length)
    offset_ndx = (raw_offset_ndx + onset_trim) * args.hop_length + \
        peak_samp_ndx - onset_offset
    offset = offset_ndx / fs
    modded_offset = offset % spb * 1000

    # Print and return the results
    elapsed_time = time.time() - initial_time
    if bpm <= 0:
        print(f"Failed to get BPM: {rounded_bpm}")
    else:
        print("Completed!")
        print("- Beats Per Minute: %d" % rounded_bpm)
        print("- BPM Error: %.2f" % rel_err)
        print("- Offset: %.1fms" % modded_offset)
        print("- Offset Error: %.1fms" % (args.hop_length / fs * 1000))
        print("- Run Time: %.1fs" % elapsed_time)
    return bpm, modded_offset
