import os
import numpy as np
import librosa
import soundfile as sf

from .utils import load_folder, plot, save
from .statistics import estimate_ibi

def fill_missing_beats(annotation: np.ndarray, end: float = 0) -> np.ndarray:
    """Interpolate missing beats based on tempo and extend to start/end.
    
    Args:
        annotation: Beat timestamps in seconds
        end: Latest timestamp to extend to (0 means don't extend end)
    
    Returns:
        Extended annotation with interpolated beats
    """
    ibi = estimate_ibi(annotation)
    if ibi == 0:
        return annotation
    
    intervals = np.diff(annotation)
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


def combine_annotations(annotations: list, voting_window: float = 0.07) -> np.ndarray:
    """Keep only beats where all annotators agree within voting_window.
    
    Args:
        annotations: List of beat timestamp arrays from different annotators
        voting_window: Max time difference to consider beats identical
    
    Returns:
        Merged beat timestamps
    """
    n = len(annotations)
    stacked = np.sort(np.hstack(annotations))
    
    window = np.lib.stride_tricks.sliding_window_view(stacked, n)
    ranges = window[:, n-1] - window[:, 0]
    candidates_mask = ranges < voting_window
    
    means = np.mean(window[:, :n], axis=1)
    
    valid = []
    for i in range(len(means)):
        if candidates_mask[i] and (not valid or valid[-1] + 0.3 <= means[i]):
            valid.append(means[i])
    
    return np.array(valid)

def apply_smoothing(annotation: np.ndarray, smoothing_size: float = 1.5) -> np.ndarray:
    """Smooth beat positions using local tempo-adjusted averaging.
    
    Args:
        annotation: Beat timestamps in seconds
        smoothing_size: Window size in multiples of inter-beat interval
    
    Returns:
        Smoothed beat timestamps
    """
    ibi = estimate_ibi(annotation)
    if ibi == 0:
        return annotation
    
    n = int(np.ceil(smoothing_size))
    padded = np.pad(annotation, n, mode='constant', constant_values=-10**6)
    window = np.lib.stride_tricks.sliding_window_view(padded, n*2 + 1)
    
    center = annotation[:, None]
    mask = np.abs(window - center) <= smoothing_size * ibi
    multiple = np.round((center - window) / ibi)
    adjusted = window + multiple * ibi
    
    masked_values = np.where(mask, adjusted, np.nan)
    result = np.nanmean(masked_values, axis=1)
    return result

def filter_silence(raw: np.ndarray, annotation: np.ndarray) -> np.ndarray:
    """Remove beats from sections where no annotator marked beats.
    
    Args:
        raw: Stacked unprocessed annotation arrays
        annotation: Processed beat timestamps to filter
    
    Returns:
        Filtered beat timestamps
    """
    ibi = estimate_ibi(annotation)
    window = 0.75 * ibi

    distances = np.abs(raw[:, None] - annotation)
    mask = np.any(distances < window, axis=0)
    
    return annotation[mask]

def pipeline(annotation_path: str, smoothing_size: float = 1.5, voting_window: float = 0.07, lag: float = 0.0, is_plot: bool = False) -> tuple[np.ndarray, float]:
    """Process raw annotations through fill, smooth, vote, and downbeat prediction.
    
    Args:
        annotation_path: Directory containing annotation .txt files
        smoothing_size: Smoothing window in multiples of IBI
        voting_window: Agreement threshold in seconds
        lag: Time offset to apply to all beats
        is_plot: Whether to display plot
    
    Returns:
        Beat timestamps with downbeat positions, percentage of real annotations
    """
    raw_annotations, raw_measures = load_folder(annotation_path)
    stacked = np.hstack(raw_annotations)
    annotations = []
    for annotation in raw_annotations:
        annotation = fill_missing_beats(annotation)
        annotation = apply_smoothing(annotation, smoothing_size)
        annotation = filter_silence(stacked, annotation)
        annotations.append(annotation)

    result = combine_annotations(annotations, voting_window)
    ibi = estimate_ibi(result)
    length = len(result)
    if length <= 1 or ibi == 0:
        result = annotations[0]
    result = fill_missing_beats(result, max(stacked))
    result = apply_smoothing(result, smoothing_size)
    
    ibi = estimate_ibi(result)
    bpm = 60 / ibi
    beats_in_bar = 4
    
    offset = (raw_measures[0][0] - np.round(raw_annotations[0][0] / ibi) - 1) % beats_in_bar
    first_beat_index = int(round(result[0] / ibi) + offset)
    beat_positions = ((first_beat_index + np.arange(len(result))) % beats_in_bar) + 1
    
    filtered_result = filter_silence(stacked, result)
    mask = np.isin(result, filtered_result)
    result = result[mask]
    beat_positions = beat_positions[mask]
    result = np.maximum(result + lag, 0)
    result = np.column_stack([result, beat_positions])

    real = length / result.shape[0]
    if is_plot:
        title = f"real: {real * 100:.2f}%, bpm: {bpm:.2f}"
        plot(stacked, result, title=title)
    return result, real

def write_sample(audio_path: str, dataset_path: str, annotation: np.ndarray, file_name: str, cutoff: float = 2.0):
    """Save audio and annotation files to dataset directory.
    
    Args:
        audio_path: Path to source audio file
        dataset_path: Directory to save dataset files
        annotation: 2D array with timestamps and beat positions
        file_name: Output filename without extension
        cutoff: Seconds of audio to keep after last beat
    """
    y, sr = librosa.load(audio_path, sr=None)
    end = annotation[-1, 0]
    y = y[:int((end + cutoff)*sr)]
    
    annotation_path = os.path.join(dataset_path, f'{file_name}.beats')
    save(annotation, annotation_path)

    audio_filename = os.path.join(dataset_path, f'{file_name}.wav')
    sf.write(audio_filename, y, sr)


def create_dataset(
        dataset_path: str, annotation_path: str,
        smoothing_size: float = 1.5, voting_window: float = 0.07, 
        lag: float = 0.0,
        threshold: float = 0.5,
        cutoff: float = 2.0, 
    ):
    """Process all annotations and create dataset with audio files.
    
    Args:
        dataset_path: Output directory for dataset
        annotation_path: Directory containing annotation subfolders
        smoothing_size: Smoothing window in multiples of IBI
        voting_window: Agreement threshold in seconds
        cutoff: Seconds of audio to keep after last beat
        threshold: Minimum ratio of real annotations to include sample
    """
    os.makedirs(dataset_path, exist_ok=True)
    folders = os.listdir(annotation_path)
    folders.sort(key=lambda x: (len(x), x))
    for subfolder in folders:
        subfolder_path = os.path.join(annotation_path, subfolder)
        if not os.path.isdir(subfolder_path):
            continue
        
        files = [f for f in os.listdir(subfolder_path) if f.endswith(('.wav', '.mp3'))]
        
        if not files:
            continue
        
        audio_file = os.path.join(subfolder_path, files[0])
        
        annotation, real = pipeline(subfolder_path, smoothing_size, voting_window, lag, is_plot=False)
        if real < threshold:
            continue
        dataset_name = os.path.basename(os.path.normpath(dataset_path))
        name = f"{dataset_name}{subfolder}"
        write_sample(audio_file, dataset_path, annotation, name, cutoff)