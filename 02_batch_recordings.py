# -*- coding: utf-8 -*-
"""
===============================================================
BATCH PING-PONG BALL IMPACT ANALYSIS
===============================================================

PURPOSE
-------
This program automatically processes all Excel recordings
stored in five release-height folders.

INPUT DIRECTORY STRUCTURE
-------------------------

    data/
        10cm/
            recording_01.xlsx
            recording_02.xlsx
            ...
        20cm/
            recording_01.xlsx
            ...
        30cm/
            ...
        40cm/
            ...
        50cm/
            ...

The program searches recursively for Excel files in these
five folders.

INPUT ROOT
----------
    data/

OUTPUT ROOT
-----------
    batch_output/

The output directory preserves the original folder structure.

For example:

    batch_output/
        10cm/
            recording_01/
                impact_detection.png
                impact_times_and_intervals.csv
                detection_summary.txt
            recording_02/
                ...
        20cm/
            ...
        30cm/
            ...
        40cm/
            ...
        50cm/
            ...

GLOBAL OUTPUT FILES
-------------------
The following files are also generated in batch_output/:

1. all_impact_times_and_intervals.csv
   A combined table containing the impact times and intervals
   from every successfully processed recording.

2. processing_summary.csv
   A summary of the processing results for each Excel file,
   including the number of detected impacts.

3. failed_files.csv
   A list of files that could not be processed, together with
   the corresponding error messages.

HOW TO RUN
----------
1. Install the required packages:

   pip install numpy pandas matplotlib scipy openpyxl

2. Make sure the data/ folder is in the same working directory
   as this Python script.

3. Run the script directly from Spyder, Jupyter, or Python IDE.

4. No command-line arguments are required.

5. No file paths need to be entered manually when running.

IMPORTANT
---------
The input root and output root are explicitly defined in the
USER SETTINGS section.

The program processes only the five specified height folders:
10cm, 20cm, 30cm, 40cm, and 50cm.

Existing output files with the same names may be overwritten.

Automatically detected impacts should be checked against the
waveforms, particularly for weak late-stage impacts.
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

INPUT_ROOT = BASE_DIR / "data"

OUTPUT_ROOT = BASE_DIR / "batch_output"

HEIGHT_FOLDERS = [
    "10cm",
    "20cm",
    "30cm",
    "40cm",
    "50cm"
]

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

    df = pd.read_excel(file_path)

    numeric = df.select_dtypes(include=[np.number])

    if numeric.empty:
        raise ValueError("No numeric audio-data column was found.")

    signal = numeric.iloc[:, 0].dropna().to_numpy(dtype=float)

    if len(signal) == 0:
        raise ValueError("The selected audio column is empty.")

    return signal


def preprocess_signal(signal):

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

    min_distance = max(1, int(MIN_PEAK_DISTANCE_S * FS))

    peaks, properties = find_peaks(
        envelope,
        height=HEIGHT_THRESHOLD,
        prominence=PROMINENCE,
        distance=min_distance
    )

    return peaks


def calculate_intervals(peaks):

    times = peaks / FS

    intervals = np.full(len(times), np.nan)

    if len(times) >= 2:
        intervals[:-1] = np.diff(times)

    return pd.DataFrame({
        "impact_index": np.arange(1, len(times) + 1),
        "impact_time_s": times,
        "interval_to_next_impact_s": intervals
    })


def save_plot(signal, envelope, peaks, output_file, title):

    time = np.arange(len(signal)) / FS

    fig, axes = plt.subplots(
        2, 1,
        figsize=(13, 8),
        sharex=True
    )

    axes[0].plot(
        time,
        signal,
        linewidth=0.45,
        color="0.45"
    )

    axes[0].set_ylabel("Audio amplitude")
    axes[0].set_title("Original audio waveform")
    axes[0].grid(alpha=0.25)

    axes[1].plot(
        time,
        envelope,
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

        for peak in peaks:
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

    fig.suptitle(title)
    fig.tight_layout()

    fig.savefig(output_file, dpi=FIGURE_DPI)
    plt.close(fig)


def process_one_file(file_path, relative_path):

    print("\nProcessing:", relative_path)

    signal = load_audio_excel(file_path)

    filtered, envelope = preprocess_signal(signal)

    peaks = detect_impacts(envelope)

    results = calculate_intervals(peaks)

    # Preserve the original height-folder structure.
    relative_folder = relative_path.parent

    # Each recording gets its own output folder.
    recording_name = relative_path.stem

    output_dir = OUTPUT_ROOT / relative_folder / recording_name

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save impact times and intervals.
    csv_file = output_dir / "impact_times_and_intervals.csv"

    results.to_csv(csv_file, index=False)

    # Save waveform and detection plot.
    plot_file = output_dir / "impact_detection.png"

    save_plot(
        signal,
        envelope,
        peaks,
        plot_file,
        title=relative_path.name
    )

    # Save individual summary.
    summary_file = output_dir / "detection_summary.txt"

    with open(summary_file, "w", encoding="utf-8") as f:

        f.write("BATCH IMPACT ANALYSIS\n")
        f.write("=" * 40 + "\n")
        f.write(f"Input file: {file_path}\n")
        f.write(f"Sampling frequency: {FS} Hz\n")
        f.write(f"Number of samples: {len(signal)}\n")
        f.write(f"Recording duration: {len(signal)/FS:.4f} s\n")
        f.write(f"Number of detected impacts: {len(peaks)}\n")

        if len(peaks) >= 2:
            f.write(
                f"Mean inter-impact interval: "
                f"{np.mean(np.diff(peaks)/FS):.6f} s\n"
            )

    # Add file identification columns to the combined table.
    results.insert(0, "recording", relative_path.stem)
    results.insert(0, "height_folder", relative_path.parent.name)
    results.insert(0, "source_file", str(relative_path))

    summary = {
        "source_file": str(relative_path),
        "height_folder": relative_path.parent.name,
        "recording": relative_path.stem,
        "number_of_samples": len(signal),
        "duration_s": len(signal) / FS,
        "number_of_impacts": len(peaks),
        "mean_interval_s": (
            float(np.mean(np.diff(peaks) / FS))
            if len(peaks) >= 2 else np.nan
        ),
        "status": "Success",
        "error": ""
    }

    print("Detected impacts:", len(peaks))

    return results, summary


# =============================================================
# MAIN PROGRAM
# =============================================================

def main():

    print("=" * 65)
    print("BATCH PING-PONG BALL IMPACT ANALYSIS")
    print("=" * 65)

    print("Input root:", INPUT_ROOT)
    print("Output root:", OUTPUT_ROOT)

    if not INPUT_ROOT.exists():
        raise FileNotFoundError(
            f"Input directory not found:\n{INPUT_ROOT}\n\n"
            "Please check that the data folder is located in "
            "the current working directory."
        )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    all_results = []
    summaries = []
    failures = []

    total_files = 0

    for height_folder in HEIGHT_FOLDERS:

        folder = INPUT_ROOT / height_folder

        if not folder.exists():
            print(f"\nWARNING: Folder not found: {folder}")
            continue

        excel_files = sorted(
            list(folder.rglob("*.xlsx")) +
            list(folder.rglob("*.xls"))
        )

        # Ignore temporary Excel lock files.
        excel_files = [
            f for f in excel_files
            if not f.name.startswith("~$")
        ]

        print(
            f"\n{height_folder}: "
            f"{len(excel_files)} Excel files found."
        )

        for file_path in excel_files:

            total_files += 1

            relative_path = file_path.relative_to(INPUT_ROOT)

            try:

                results, summary = process_one_file(
                    file_path,
                    relative_path
                )

                all_results.append(results)
                summaries.append(summary)

            except Exception as error:

                print("ERROR:", error)

                failures.append({
                    "source_file": str(relative_path),
                    "error": str(error)
                })

                summaries.append({
                    "source_file": str(relative_path),
                    "height_folder": relative_path.parent.name,
                    "recording": relative_path.stem,
                    "number_of_samples": np.nan,
                    "duration_s": np.nan,
                    "number_of_impacts": np.nan,
                    "mean_interval_s": np.nan,
                    "status": "Failed",
                    "error": str(error)
                })

    # ---------------------------------------------------------
    # Save combined impact-time and interval table.
    # ---------------------------------------------------------

    if all_results:

        combined = pd.concat(
            all_results,
            ignore_index=True
        )

        combined_file = (
            OUTPUT_ROOT / "all_impact_times_and_intervals.csv"
        )

        combined.to_csv(combined_file, index=False)

    # ---------------------------------------------------------
    # Save processing summary.
    # ---------------------------------------------------------

    summary_file = OUTPUT_ROOT / "processing_summary.csv"

    pd.DataFrame(summaries).to_csv(
        summary_file,
        index=False
    )

    # ---------------------------------------------------------
    # Save failed-file list.
    # ---------------------------------------------------------

    failure_file = OUTPUT_ROOT / "failed_files.csv"

    pd.DataFrame(
        failures,
        columns=["source_file", "error"]
    ).to_csv(failure_file, index=False)

    # ---------------------------------------------------------
    # Print final report.
    # ---------------------------------------------------------

    successful = sum(
        1 for item in summaries
        if item["status"] == "Success"
    )

    print("\n" + "=" * 65)
    print("BATCH PROCESSING COMPLETED")
    print("=" * 65)

    print("Total Excel files found:", total_files)
    print("Successfully processed:", successful)
    print("Failed files:", len(failures))

    print("\nOutput directory:", OUTPUT_ROOT)
    print("Processing summary:", summary_file)
    print("Failed-file list:", failure_file)

    if all_results:
        print(
            "Combined impact data:",
            OUTPUT_ROOT / "all_impact_times_and_intervals.csv"
        )


if __name__ == "__main__":
    main()