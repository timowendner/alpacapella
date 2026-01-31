"""Process and merge multiple beat annotations from music files.

Combines annotations from multiple annotators using voting and smoothing,
then creates a unified dataset for beat tracking models.
"""

import os
import numpy as np
import librosa
import soundfile as sf

from .utils import load_folder, plot

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


def combine_annotations(annotations: list, voting_window: float = 0.05) -> np.ndarray:
    """Merge annotations by keeping beats where all annotators agree.
    
    Beats are considered the same if within voting_window seconds.
    Prevents duplicates by filtering beats closer than 0.3 seconds.

    Args:
        annotations: List of beat timestamp arrays from different annotators
        voting_window: Max time difference to consider beats identical (typically 0.03-0.1)
    Returns:
        merged beat timestamps
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
    return np.array(result)

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

def filter_silence(raw: np.ndarray, annotation: np.ndarray) -> np.ndarray:
    """Remove beats from silent sections where no annotator marked beats.
    
    Keeps beats only if any raw annotation exists within 2 inter-beat intervals.

    Args:
        raw: stacked (unprocessed) annotation arrays
        annotation: Processed beat timestamps to filter
    Returns:
        Filtered beat timestamps with silence sections removed
    """
    beat = estimate_ibi(annotation)
    window = 0.75 * beat
    
    result = []
    for value in annotation:
        distances = np.abs(raw - value)
        if np.any(distances < window):
            result.append(value)
    return np.array(result)

def pipeline(annotation_path: str, smoothing_size: float = 2.2, voting_window: float = 0.05) -> tuple[np.ndarray, float]:
    """Complete processing pipeline from raw annotations to final merged output.
    
    Steps: load -> fill gaps -> smooth -> vote -> smooth again -> fill -> predict downbeats -> filter silence
    Displays plot and prints statistics (interpolation %, BPM).

    Args:
        annotation_path: Directory containing annotation .txt files
        smoothing_size: Smoothing window in multiples of IBI (default 2.2)
        voting_window: Agreement threshold in seconds (default 0.05)
    Returns:
        Final merged and processed beat timestamps with downbeat positions,
        percentage of real annotations (without interpolations)
    """
    raw_annotations = load_folder(annotation_path)
    stacked = np.sort(np.hstack(raw_annotations))
    annotations = []
    for annotation in raw_annotations:
        annotation = fill_missing_beats(annotation)
        annotation = apply_smoothing(annotation, smoothing_size)
        annotation = filter_silence(stacked, annotation)
        annotations.append(annotation)

    result = combine_annotations(annotations, voting_window)
    length = len(result)
    result = fill_missing_beats(result, max(stacked))
    result = apply_smoothing(result, smoothing_size)
    
    ibi = estimate_ibi(result)
    bpm = 60 / ibi
    beats_in_bar = 4 if bpm < 110 else 8
    
    first_beat_index = int(round(result[0] / ibi))
    beat_positions = ((first_beat_index + np.arange(len(result))) % beats_in_bar) + 1
    
    filtered_result = filter_silence(stacked, result)
    mask = np.isin(result, filtered_result)
    result = result[mask]
    beat_positions = beat_positions[mask]
    result = np.maximum(result, 0)
    result = np.column_stack([result, beat_positions])

    real = length / result.shape[0]
    title = f"interpolated: {(1 - real) * 100:.2f}%, bpm: {bpm:.2f}"
    plot(stacked, result, title=title)
    return result, real

def write_dataset(audio_path: str, dataset_path: str, annotation: np.ndarray, cutoff: float = 2.0):
    """Save audio and annotation to dataset with automatic numbering.
    
    Creates files: audioN.wav and annotationN.txt where N auto-increments.

    Args:
        audio_path: Path to source audio file
        dataset_path: Directory to save dataset files
        annotation: 2D array with timestamps and beat positions
        cutoff: Seconds of audio to keep after last beat (default 2.0)
    """
    y, sr = librosa.load(audio_path, sr=None)
    end = annotation[-1, 0]
    y = y[:int((end + cutoff)*sr)]
    
    existing_files = [f for f in os.listdir(dataset_path) if f.startswith('audio')]
    next_num = len(existing_files) + 1
    
    annotation_filename = os.path.join(dataset_path, f'annotation{next_num}.beats')
    with open(annotation_filename, 'w') as f:
        for t, beat_pos in annotation:
            f.write(f"{t:.9f} {int(beat_pos)}\n")

    audio_filename = os.path.join(dataset_path, f'audio{next_num}.wav')
    sf.write(audio_filename, y, sr)


def create_dataset(
        dataset_path: str, annotation_path: str, 
        smoothing_size: float = 2.2, voting_window: float = 0.05, 
        cutoff: float = 2.0, beats_in_bar: int = 4
    ):
    os.makedirs(dataset_path, exist_ok=True)
    for subfolder in os.listdir(annotation_path):
        subfolder_path = os.path.join(annotation_path, subfolder)
        if not os.path.isdir(subfolder_path):
            continue
        
        wav_files = [f for f in os.listdir(subfolder_path) if f.endswith('.wav')]
        
        if not wav_files:
            print(f"No .wav file in {subfolder}, skipping")
            continue
        
        audio_file = os.path.join(subfolder_path, wav_files[0])
        
        annotation, real = pipeline(subfolder_path, smoothing_size, voting_window)
        write_dataset(audio_file, dataset_path, annotation, beats_in_bar, cutoff)