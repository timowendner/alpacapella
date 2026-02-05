import soundfile as sf
import numpy as np
import os

from .utils import load

def estimate_ibi(annotation: np.ndarray) -> float:
    """Estimate inter-beat interval using median of valid intervals (60-200 BPM).

    Args:
        annotation: Beat timestamps in seconds
    
    Returns:
        Median inter-beat interval in seconds
    """
    intervals = np.diff(annotation)
    intervals = intervals[intervals > 0.3]
    intervals = intervals[intervals < 1]
    if len(intervals) == 0:
        return 0
    ibi = np.median(intervals)
    return float(ibi)


def statistics(folder_path: str):
    """Collect beat counts, BPMs, and audio lengths from all files in folder.

    Args:
        folder_path: Directory to search recursively
    
    Returns:
        Lists of beat counts, BPMs, and audio durations
    """
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