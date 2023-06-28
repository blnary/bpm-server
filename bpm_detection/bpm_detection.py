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
import array
import math
from pydub import AudioSegment

import matplotlib.pyplot as plt
import numpy
import pywt
from scipy import signal
from tqdm import tqdm


def read_mp3(filename):
    # Load the MP3 file
    try:
        audio = AudioSegment.from_mp3(filename)
    except Exception as e:
        print(e)
        return None, None

    # Extract audio data and sample rate
    samps = array.array("i", audio.raw_data)
    fs = audio.frame_rate

    return samps, fs


# print an error when no data can be found
def no_audio_data():
    print("No audio data for sample, skipping...")
    return None, None


# simple peak detection
def peak_detect(data):
    return numpy.argmax(data)


def bpm_detector(data, fs):
    cA = []
    cD = []
    correl = []
    cD_sum = []
    levels = 4
    max_decimation = 2 ** (levels - 1)
    min_ndx = math.floor(60.0 / 220 * (fs / max_decimation))
    max_ndx = math.floor(60.0 / 40 * (fs / max_decimation))

    for loop in range(0, levels):
        cD = []
        # 1) DWT
        if loop == 0:
            [cA, cD] = pywt.dwt(data, "db4")
            cD_minlen = len(cD) / max_decimation + 1
            cD_sum = numpy.zeros(math.floor(cD_minlen))
        else:
            [cA, cD] = pywt.dwt(cA, "db4")

        # 2) Filter
        cD = signal.lfilter([0.01], [1 - 0.99], cD)[0: math.floor(cD_minlen)]

        # 4) Subtract out the mean.

        # 5) Decimate for reconstruction later.
        cD = abs(cD)
        cD = cD - numpy.mean(cD)

        # 6) Recombine the signal before ACF
        #    Essentially, each level the detail coefs (i.e. the HPF values)
        #    are concatenated to the beginning of the array
        cD_sum = cD + cD_sum

    if [b for b in cA if b != 0.0] == []:
        return no_audio_data()

    # Adding in the approximate data as well...
    cA = signal.lfilter([0.01], [1 - 0.99], cA)[0: math.floor(cD_minlen)]
    cA = abs(cA)
    cA = cA - numpy.mean(cA)
    cD_sum = cA + cD_sum

    # ACF
    correl = numpy.correlate(cD_sum, cD_sum, "full")

    midpoint = len(correl) // 2
    correl_midpoint_tmp = correl[midpoint + min_ndx:midpoint + max_ndx]
    mult = numpy.arange(min_ndx, max_ndx) / max_ndx
    correl_midpoint_tmp = correl_midpoint_tmp * mult
    correl_midpoint_tmp = correl_midpoint_tmp / \
        numpy.linalg.norm(correl_midpoint_tmp)

    # correl_midpoint_tmp = numpy.convolve(
    #     correl_midpoint_tmp, [-1, 1], mode='same')
    peak_ndx = peak_detect(correl_midpoint_tmp)
    peak_ndx_adjusted = peak_ndx + min_ndx
    bpm = 60.0 / peak_ndx_adjusted * (fs / max_decimation)

    # Get seconds per beat
    n = numpy.arange(0, len(correl_midpoint_tmp))
    n = (n + min_ndx) / (fs / max_decimation)
    plt.plot(n, correl_midpoint_tmp)
    return bpm, correl_midpoint_tmp[peak_ndx]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process .mp3 file to determine the Beats Per Minute.")
    parser.add_argument("--filename", required=True,
                        help=".mp3 file for processing")
    parser.add_argument(
        "--window",
        type=float,
        default=3,
        help="Size of the the window (seconds) that will be scanned to \
                determine the bpm. Typically less than 10 seconds. [3]",
    )

    args = parser.parse_args()
    samps, fs = read_mp3(args.filename)
    data = []
    max_correl = 0
    bpm = 0
    bpm_by_max = 0
    n = 0
    nsamps = len(samps)
    window_samps = int(args.window * fs)
    samps_ndx = 0  # First sample in window_ndx
    max_window_ndx = math.floor(nsamps / window_samps)

    # Iterate through all windows
    for window_ndx in tqdm(range(0, max_window_ndx)):

        # Get a new set of samples
        # print(n,":",len(bpms),":",max_window_ndx_int,":",fs,":",nsamps,":",samps_ndx)
        data = samps[samps_ndx: samps_ndx + window_samps]
        if not ((len(data) % window_samps) == 0):
            raise AssertionError(str(len(data)))

        bpm, correl = bpm_detector(data, fs)
        if bpm is None:
            continue

        if correl > max_correl:
            max_correl = correl
            bpm_by_max = bpm

        # Iterate at the end of the loop
        samps_ndx = samps_ndx + window_samps

        # Counter for debug...
        n = n + 1

    while bpm_by_max < 120:
        bpm_by_max *= 2
    while bpm_by_max >= 240:
        bpm_by_max /= 2
    bpm_by_max = round(bpm_by_max * 2) / 2
    print("Completed!  Estimated Beats Per Minute:", bpm_by_max)

    plt.show(block=True)
