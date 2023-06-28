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


def bpm_detector(data, fs, decimation, downsample):
    min_ndx = math.floor(60.0 / 240 * (fs / decimation))
    max_ndx = math.floor(60.0 / 40 * (fs / decimation))

    # Downsample
    if downsample:
        remainder = len(data) % decimation
        zeros_needed = decimation - remainder
        data = np.pad(data, (0, zeros_needed), mode='constant')
        data = np.mean(data.reshape(-1, decimation), axis=1)

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

    # Detect
    peak_ndx = np.argmax(correl_midpoint_tmp)
    peak_ndx_adjusted = peak_ndx + min_ndx
    bpm = 60.0 / peak_ndx_adjusted * (fs / decimation)

    # Get seconds per beat
    n = np.arange(0, len(correl_midpoint_tmp))
    n = (n + min_ndx) / (fs / decimation)
    plt.plot(n, correl_midpoint_tmp)
    return bpm, correl_midpoint_tmp[peak_ndx]


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
        "--hop_length",
        type=int,
        default=64,
        help="Hop legnth for onset strength calculation."
    )
    parser.add_argument(
        "--use_onset",
        type=bool,
        default=False,
        help="Toggle use onset information or not."
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
    hop_length = args.hop_length if args.use_onset else 1
    samps, fs = gen_sin() if args.use_sine else librosa.load(args.filename)
    if args.use_onset:
        print("Calculating onset strength...")
        samps = librosa.onset.onset_strength(
            y=samps, sr=fs, hop_length=hop_length)
    print("Doing auto correlation...")
    data = []
    max_correl = 0
    bpm = 0
    bpm_by_max = 0
    n = 0
    nsamps = len(samps)
    window_samps = int(args.window * fs / hop_length)
    samps_ndx = 0
    max_window_ndx = math.floor(nsamps / window_samps)

    # Iterate through all windows
    for window_ndx in range(0, max_window_ndx):

        # Get a new set of samples
        data = samps[samps_ndx: samps_ndx + window_samps]
        if not ((len(data) % window_samps) == 0):
            raise AssertionError(str(len(data)))

        # Detect bpm
        decimation = hop_length if args.use_onset else args.decimation
        bpm, correl = bpm_detector(
            data, fs, decimation, not args.use_onset)
        if bpm is None:
            continue

        # Use the most confident bpm
        if correl > max_correl:
            max_correl = correl
            bpm_by_max = bpm

        # Iterate at the end of the loop
        samps_ndx = samps_ndx + window_samps

        # Counter for debug...
        n = n + 1

    # Check BPM validity
    if bpm_by_max <= 0:
        print(f"Failed to get BPM: {bpm_by_max}")
    else:
        while bpm_by_max < 120:
            bpm_by_max *= 2
        while bpm_by_max >= 240:
            bpm_by_max /= 2
        print("Completed!")
        print(f"Beats Per Minute: {bpm_by_max}")

    # Plot
    plt.hlines([0.5 if args.use_onset else 0.15], 0.25, 1.5, alpha=0.5, color='r',
               linestyle='--', label='Thres')
    plt.show(block=True)
