"""Process and merge multiple beat annotations from music files.

Combines annotations from multiple annotators using voting and smoothing,
then creates a unified dataset for beat tracking models.
"""

import os
import numpy as np
import librosa
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt

def estimate_ibi(annotation: np.ndarray) -> float:
    """Estimate inter-beat interval using median of valid intervals.
    
    Filters intervals between 0.3-1.0 seconds (60-200 BPM range).

    Args:
        annotation: Array of beat timestamps in seconds
    Returns:
        Median inter-beat interval in seconds
    """
    intervals = np.diff(annotation)
    intervals = intervals[intervals > 0.3]
    intervals = intervals[intervals < 1]
    ibi = np.median(intervals)
    return float(ibi)

def fill_missing_beats(annotation: np.ndarray, end: float = 0) -> np.ndarray:
    """Add interpolated beats where annotator missed beats.
    
    Extends annotation to start (time 0) and end based on estimated tempo.

    Args:
        annotation: Array of beat timestamps in seconds
        end: Latest timestamp to extend to (0 means don't extend end)
    Returns:
        Extended annotation array with interpolated beats
    """
    intervals = np.diff(annotation)
    ibi = estimate_ibi(annotation)
    result = [annotation[0]]
    for i in range(len(intervals)):
        current = annotation[i]
        interval = intervals[i]
        missing = round(interval / ibi) - 1
        if missing > 0:
            step = interval / (missing + 1)
            for j in range(1, missing + 1):
                result.append(current + step * j)
        result.append(annotation[i + 1])

    end += ibi/2
    while end is not None and result[-1]+ibi < end:
        result.append(result[-1]+ibi)
    result.reverse()
    while result[-1] - ibi > 0:
        result.append(result[-1] - ibi)
    result.reverse()
    return np.array(result)


def load_annotations(annotation_path: str) -> list[np.ndarray]:
    """Load all annotation files from directory.
    
    Expects CSV files with 'TIME' column containing beat timestamps.

    Args:
        annotation_path: Directory containing .txt annotation files
    Returns:
        List of beat timestamp arrays, one per file
    """
    annotations = []
    for file in sorted(os.listdir(annotation_path)):
        if not file.endswith('.txt'):
            continue
        file_path = os.path.join(annotation_path, file)
        df = pd.read_csv(file_path)
        annotation = np.array(df['TIME'].values)
        annotations.append(annotation)
    return annotations

def combine_annotations(annotations: list, voting_window: float = 0.05) -> tuple[np.ndarray, int]:
    """Merge annotations by keeping beats where all annotators agree.
    
    Beats are considered the same if within voting_window seconds.
    Prevents duplicates by filtering beats closer than 0.3 seconds.

    Args:
        annotations: List of beat timestamp arrays from different annotators
        voting_window: Max time difference to consider beats identical (typically 0.03-0.1)
    Returns:
        Tuple of (merged beat timestamps, latest timestamp from any annotation)
    """
    n = len(annotations)
    stacked = np.sort(np.hstack(annotations))
    result = []
    for i, value in enumerate(stacked[:-n+1]):
        candidate = stacked[i+n-1]
        mean = np.mean(stacked[i:i+n-1])
        if result and result[-1] + 0.3 > mean:
            continue
        if candidate - value < voting_window:
            result.append(mean)
    result = np.array(result)
    return result, max(stacked)

def apply_smoothing(annotation: np.ndarray, smoothing_size: float = 3) -> np.ndarray:
    """Smooth beat positions using local tempo-adjusted averaging.
    
    Each beat is adjusted based on nearby beats within smoothing_size intervals.

    Args:
        annotation: Array of beat timestamps in seconds
        smoothing_size: Window size in multiples of inter-beat interval (typically 2-4)
    Returns:
        Smoothed beat timestamps
    """
    ibi = estimate_ibi(annotation)
    result = []
    
    for center in annotation:
        mask = np.abs(annotation - center) < smoothing_size * ibi
        candidates = annotation[mask].copy()
        multiple = np.round((center - candidates) / ibi)
        candidates += multiple * ibi
        result.append(np.mean(candidates))
    return np.array(result)

def play(audio_path: str, click: np.ndarray):
    """Play audio with beat click overlay in Jupyter notebook.
    
    Requires IPython environment. Will not work in standard Python scripts.

    Args:
        audio_path: Path to audio file
        click: Beat timestamps in seconds for click track overlay
    """
    try:
        from IPython.display import Audio, display
    except ImportError:
        raise RuntimeError("play() requires IPython (use in Jupyter notebook)")
    
    y, sr = librosa.load(audio_path, sr=None)
    clicks = librosa.clicks(times=click, sr=sr, length=len(y), click_freq=1000)
    display(Audio(y + clicks, rate=sr))

def filter_silence(raw: list, annotation: np.ndarray) -> np.ndarray:
    """Remove beats from silent sections where no annotator marked beats.
    
    Keeps beats only if any raw annotation exists within 2 inter-beat intervals.

    Args:
        raw: List of original (unprocessed) annotation arrays
        annotation: Processed beat timestamps to filter
    Returns:
        Filtered beat timestamps with silence sections removed
    """
    beat = estimate_ibi(annotation)
    window = 2 * beat
    raw_stacked = np.hstack(raw)
    
    result = []
    for value in annotation:
        distances = np.abs(raw_stacked - value)
        if np.any(distances < window):
            result.append(value)
    return np.array(result)

def pipeline(annotation_path: str, smoothing_size: float = 2.2, voting_window: float = 0.05) -> np.ndarray:
    """Complete processing pipeline from raw annotations to final merged output.
    
    Steps: load -> fill gaps -> smooth -> vote -> smooth again -> fill -> filter silence
    Displays plot and prints statistics (interpolation %, BPM).

    Args:
        annotation_path: Directory containing annotation .txt files
        smoothing_size: Smoothing window in multiples of IBI (default 2.2)
        voting_window: Agreement threshold in seconds (default 0.05)
    Returns:
        Final merged and processed beat timestamps
    """
    raw_annotations = load_annotations(annotation_path)
    annotations = []
    for annotation in raw_annotations:
        annotation = fill_missing_beats(annotation)
        annotation = apply_smoothing(annotation, smoothing_size)
        annotations.append(annotation)

    result, end = combine_annotations(annotations, voting_window)
    result = apply_smoothing(result, smoothing_size)
    length = len(result)
    stacked = np.sort(np.hstack(annotations))
    plot(stacked, result)
    result = fill_missing_beats(result, end)
    result = filter_silence(raw_annotations, result)

    interpolated = length / len(result)
    print(f"interpolated: {(1 - interpolated) * 100:.2f}%")
    print(f"bpm: {60 / np.mean(np.diff(result)):.2f}")
    return result

def plot(raw: np.ndarray, final: np.ndarray):
    """Visualize raw and final beat annotations.
    
    Blue lines: all input beats, Red lines: final merged beats.

    Args:
        raw: Combined timestamps from all raw annotations
        final: Final processed beat timestamps
    """
    plt.figure(figsize=(10, 2))
    plt.vlines(x=raw, ymin=0, ymax=1, linewidth=0.4)
    plt.vlines(x=final, ymin=0, ymax=1, linewidth=0.4, colors='red')
    plt.show()

def write_dataset(audio_path: str, dataset_path: str, annotation: np.ndarray, beats_in_bar: int = 4, cutoff: float = 2.0):
    """Save audio and annotation to dataset with automatic numbering.
    
    Creates files: audioN.wav and annotationN.txt where N auto-increments.

    Args:
        audio_path: Path to source audio file
        dataset_path: Directory to save dataset files
        annotation: Beat timestamps in seconds
        beats_in_bar: beats in a Bar. Normal is 4/4.
        cutoff: Seconds of audio to keep after last beat (default 2.0)
    """
    y, sr = librosa.load(audio_path, sr=None)
    end = annotation[-1]
    y = y[:int((end + cutoff)*sr)]
    
    existing_files = [f for f in os.listdir(dataset_path) if f.startswith('audio')]
    next_num = len(existing_files) + 1
    
    ibi = estimate_ibi(annotation)
    labels = []
    for t in annotation:
        total_beats = int(round(t / ibi))
        bar = (total_beats // beats_in_bar) + 1
        beat_in_bar = (total_beats % beats_in_bar) + 1
        labels.append(f"{bar}.{beat_in_bar}")
    
    df = pd.DataFrame({'TIME': annotation, 'LABEL': labels})
    annotation_filename = os.path.join(dataset_path, f'annotation{next_num}.txt')
    df.to_csv(annotation_filename, index=False)

    audio_filename = os.path.join(dataset_path, f'audio{next_num}.wav')
    sf.write(audio_filename, y, sr)