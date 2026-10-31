# -*- coding: utf-8 -*-
"""
===============================================================
SINGLE-FILE PING-PONG BALL IMPACT ANALYSIS
===============================================================

PURPOSE
-------
This program analyzes one Excel file containing raw audio data
recorded during the bouncing of a ping-pong ball.

INPUT FILE
----------
Original Audio 2026-10-26 06-46-19.xlsx

The input file must be located in the same directory as this
Python script. If it is stored elsewhere, change INPUT_FILE
below to its full path.

OUTPUT
------
All results are automatically saved in the folder:

    single_output/

The program generates:

1. impact_detection.png
   Audio waveform and detected impact-event locations.

2. impact_times_and_intervals.csv
   Detected impact times and intervals between consecutive
   impacts.

3. detection_summary.txt
   Basic information about the recording and detected events.

HOW TO RUN
----------
1. Install the required packages if necessary:

   pip install numpy pandas matplotlib scipy openpyxl

2. Put this script and the input Excel file in the same folder.

3. Run this script directly from Spyder, Jupyter, or Python IDE.

4. No command-line arguments or interactive input are required.

NOTES
-----
The detection parameters are defined in the USER SETTINGS
section below. They can be adjusted if weak late-stage impacts
are missed or false events are detected.

The automatically detected events should be checked against
the original audio waveform before using the results in a
scientific analysis.
===============================================================
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt


# =============================================================
# USER SETTINGS: EDIT PATHS HERE IF NECESSARY
# =============================================================

BASE_DIR = Path.cwd()

INPUT_FILE = BASE_DIR / "data/30cm/Original Audio 2026-10-26 06-46-19.xlsx"

OUTPUT_DIR = BASE_DIR / "single_output"

# Audio sampling frequency
FS = 16000

# Detection parameters
MIN_PEAK_DISTANCE_S = 0.08
PROMINENCE = 0.008
HEIGHT_THRESHOLD = 0.003

# Signal-processing parameters
ENVELOPE_WINDOW_S = 0.003
FILTER_LOW_HZ = 80
FILTER_HIGH_HZ = 7000

# Plot settings
FIGURE_DPI = 300


# =============================================================
# FUNCTIONS
# =============================================================

def load_audio_excel(file_path):
    """Load the first usable numeric column from an Excel file."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"Input Excel file not found:\n{file_path}\n\n"
            "Please check INPUT_FILE in the USER SETTINGS section."
        )

    df = pd.read_excel(file_path)

    numeric = df.select_dtypes(include=[np.number])

    if numeric.empty:
        raise ValueError("No numeric audio-data column was found.")

    # Use the first numeric column as the audio signal.
    signal = numeric.iloc[:, 0].dropna().to_numpy(dtype=float)

    if len(signal) == 0:
        raise ValueError("The selected audio column is empty.")

    return signal


def preprocess_signal(signal):
    """Remove DC offset and calculate a smoothed amplitude envelope."""

    signal = np.asarray(signal, dtype=float)
    signal = signal - np.mean(signal)

    nyquist = FS / 2

    low = FILTER_LOW_HZ / nyquist
    high = FILTER_HIGH_HZ / nyquist

    if 0 < low < high < 1:
        b, a = butter(3, [low, high], btype="bandpass")
        filtered = filtfilt(b, a, signal)
    else:
        filtered = signal.copy()

    window = max(1, int(ENVELOPE_WINDOW_S * FS))

    envelope = np.convolve(
        np.abs(filtered),
        np.ones(window) / window,
        mode="same"
    )

    return filtered, envelope


def detect_impacts(envelope):
    """Detect impact events from the amplitude envelope."""

    min_distance = max(1, int(MIN_PEAK_DISTANCE_S * FS))

    peaks, properties = find_peaks(
        envelope,
        height=HEIGHT_THRESHOLD,
        prominence=PROMINENCE,
        distance=min_distance
    )

    return peaks, properties


def calculate_intervals(peaks):
    """Calculate impact times and intervals between impacts."""

    times = peaks / FS

    intervals = np.full(len(times), np.nan)

    if len(times) >= 2:
        intervals[:-1] = np.diff(times)

    result = pd.DataFrame({
        "impact_index": np.arange(1, len(times) + 1),
        "impact_time_s": times,
        "interval_to_next_impact_s": intervals
    })

    return result


def save_plot(signal, filtered, envelope, peaks, output_file):
    """Save waveform, envelope, and detected impact locations."""

    time = np.arange(len(signal)) / FS

    fig, axes = plt.subplots(
        2, 1,
        figsize=(13, 8),
        sharex=True
    )

    axes[0].plot(
        time, signal,
        linewidth=0.45,
        color="0.45"
    )

    axes[0].set_ylabel("Audio amplitude")
    axes[0].set_title("Original audio waveform")
    axes[0].grid(alpha=0.25)

    axes[1].plot(
        time, envelope,
        linewidth=0.8,
        color="tab:blue",
        label="Amplitude envelope"
    )

    if len(peaks) > 0:
        axes[1].scatter(
            peaks / FS,
            envelope[peaks],
            color="tab:red",
            s=30,
            zorder=3,
            label="Detected impacts"
        )

        for i, peak in enumerate(peaks):
            axes[1].axvline(
                peak / FS,
                color="tab:red",
                alpha=0.25,
                linewidth=0.8
            )

    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Envelope amplitude")
    axes[1].set_title("Impact-event detection")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_file, dpi=FIGURE_DPI)
    plt.close(fig)


# =============================================================
# MAIN PROGRAM
# =============================================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SINGLE-FILE IMPACT ANALYSIS")
    print("=" * 60)

    print("Input:", INPUT_FILE)
    print("Output:", OUTPUT_DIR)

    signal = load_audio_excel(INPUT_FILE)

    filtered, envelope = preprocess_signal(signal)

    peaks, properties = detect_impacts(envelope)

    results = calculate_intervals(peaks)

    # Save CSV
    csv_file = OUTPUT_DIR / "impact_times_and_intervals.csv"
    results.to_csv(csv_file, index=False)

    # Save plot
    plot_file = OUTPUT_DIR / "impact_detection.png"

    save_plot(
        signal,
        filtered,
        envelope,
        peaks,
        plot_file
    )

    # Save summary
    summary_file = OUTPUT_DIR / "detection_summary.txt"

    with open(summary_file, "w", encoding="utf-8") as f:

        f.write("SINGLE-FILE IMPACT ANALYSIS\n")
        f.write("=" * 40 + "\n")
        f.write(f"Input file: {INPUT_FILE}\n")
        f.write(f"Sampling frequency: {FS} Hz\n")
        f.write(f"Number of samples: {len(signal)}\n")
        f.write(f"Recording duration: {len(signal)/FS:.4f} s\n")
        f.write(f"Number of detected impacts: {len(peaks)}\n")

        if len(peaks) >= 2:
            f.write(
                f"Mean inter-impact interval: "
                f"{np.mean(np.diff(peaks)/FS):.6f} s\n"
            )

    print("\nAnalysis completed.")
    print("Detected impacts:", len(peaks))
    print("CSV:", csv_file)
    print("Plot:", plot_file)
    print("Summary:", summary_file)


if __name__ == "__main__":
    main()