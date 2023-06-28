# Copyright 2012 Free Software Foundation, Inc.
#
# This file is part of The BPM Detector Python
#
# The BPM Detector Python is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3, or (at your option)
# any later version.
#
# The BPM Detector Python is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with The BPM Detector Python; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street,
# Boston, MA 02110-1301, USA.

import argparse
import librosa
import math

import matplotlib.pyplot as plt
import numpy as np


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
    for i in range(1, 10):
        for j in range(1, 10):
            coef = i / j
            rel_err = 0
            unit = spb_peak * coef
            if 60 / unit > 240 or 60 / unit < 120:
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


def bpm_detector(data, fs, decimation):
    min_ndx = math.floor(60.0 / 240 * (fs / decimation))
    max_ndx = math.floor(60.0 / 40 * (fs / decimation))

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
    high_ndx = np.argwhere(correl_midpoint_tmp > 0.15)
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
    plt.plot(n, correl_midpoint_tmp)
    return spb_candidate, spb_peak, correl


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process audio file to determine the Beats Per Minute.")
    parser.add_argument("--filename", required=True,
                        help="audio file for processing")
    parser.add_argument(
        "--window",
        type=float,
        default=3,
        help="Size of the the window (seconds) that will be scanned to \
                determine the bpm. Typically less than 10 seconds. [3]",
    )
    parser.add_argument(
        "--decimation",
        type=int,
        default=16,
        help="Downsample decimation, higher value means less processing time."
    )
    parser.add_argument(
        "--use_sine",
        type=bool,
        default=False,
        help="Use template sine wave as audio input."
    )
    args = parser.parse_args()

    # Preprocess
    print("Loading file...")
    samps, fs = gen_sin() if args.use_sine else librosa.load(args.filename)
    print("Doing auto correlation...")
    data = []
    local_spb_candidate = 0
    peak_correl = 0
    peak_spb = 0
    spb_candidate = np.array([], dtype=np.float64)
    bpm = 0
    n = 0
    nsamps = len(samps)
    window_samps = int(args.window * fs)
    samps_ndx = 0
    max_window_ndx = math.floor(nsamps / window_samps)

    # Iterate through all windows
    for window_ndx in range(0, max_window_ndx):

        # Get a new set of samples
        data = samps[samps_ndx: samps_ndx + window_samps]
        if not ((len(data) % window_samps) == 0):
            raise AssertionError(str(len(data)))

        # Detect bpm
        local_spb_candidate, local_spb_peak, correl = bpm_detector(
            data, fs, args.decimation)

        # Remember candidates and peak
        if local_spb_candidate is None:
            continue
        spb_candidate = np.append(spb_candidate, local_spb_candidate)
        if correl > peak_correl:
            peak_correl = correl
            peak_spb = local_spb_peak

        # Iterate at the end of the loop
        samps_ndx = samps_ndx + window_samps

        # Counter for debug...
        n = n + 1

    # Calculate BPM by GCD
    spb = gcd_spb(spb_candidate, peak_spb)
    bpm = 60 / spb
    rounded_bpm = round(bpm)
    rel_err = bpm - rounded_bpm

    # Check BPM validity
    if bpm <= 0:
        print(f"Failed to get BPM: {rounded_bpm}")
    else:
        print("Completed!")
        print(f"Beats Per Minute: {rounded_bpm}\nRelative Error: {rel_err}")

    # Plot
    vlines = np.arange(10) * spb
    vlines = vlines[vlines > 0.25]
    vlines = vlines[vlines < 1.5]
    plt.hlines([0.15], 0.25, 1.5, alpha=0.5, color='r',
               linestyle='--', label='Thres')
    plt.vlines(vlines, 0.15, peak_correl, alpha=0.5, color='b',
               linestyle='--', label='Beat')
    plt.show(block=True)
