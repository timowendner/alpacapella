import soundfile as sf
import numpy as np
import os

from .utils import load

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


def statistics(folder_path: str):
    beats = []
    bpms = []
    audio_length = []
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            
            if file.endswith('.txt') or file.endswith('.beats'):
                annotation = load(file_path)
                beats.append(len(annotation))
                ibi = estimate_ibi(annotation[:, 0])
                bpms.append(60 / ibi)
            
            elif file.endswith('.wav') or file.endswith('.mp3'):
                info = sf.info(file_path)
                audio_length.append(info.duration)

    return (
        beats,
        bpms,
        audio_length
    )