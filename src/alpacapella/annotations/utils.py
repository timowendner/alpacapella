import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import mir_eval


def load(annotation_path: str) -> tuple[np.ndarray]:
    """Load an annotation file.

    Args:
        annotation_path: File containing annotations
    Returns:
        Beat timestamp and downbeat array
    """
    annotation = np.loadtxt(annotation_path)
    return annotation


def load_folder(annotation_path: str) -> list[np.ndarray]:
    """Load all annotation files from directory.

    Args:
        annotation_path: Directory containing annotation files
    Returns:
        List of beat timestamp arrays, one per file
    """
    annotations = []
    for file in sorted(os.listdir(annotation_path)):
        if not file.endswith('.txt') or file.endswith('.beats'):
            continue
        file_path = os.path.join(annotation_path, file)
        annotation = load(file_path)[:, 0]
        annotations.append(annotation)
    return annotations


def plot(raw: np.ndarray, annotation: np.ndarray, title: str):
    """Visualize raw and final beat annotations.
    
    Blue lines: all input beats, Red lines: downbeats, Orange lines: other beats.

    Args:
        raw: Combined timestamps from all raw annotations
        annotation: 2D array with timestamps and beat positions
        title: the title of the plot
    """
    plt.figure(figsize=(10, 2))
    plt.vlines(x=raw, ymin=0, ymax=1, linewidth=0.4)
    
    downbeats = annotation[annotation[:, 1] == 1, 0]
    other_beats = annotation[annotation[:, 1] != 1, 0]
    
    plt.vlines(x=downbeats, ymin=0, ymax=1, linewidth=0.4, colors='red')
    plt.vlines(x=other_beats, ymin=0, ymax=1, linewidth=0.4, colors='orange')
    plt.title(title)
    plt.yticks([])
    plt.show()

def play(audio_path: str, annotation: np.ndarray):
    """Play audio with beat click overlay in Jupyter notebook.
    
    Requires IPython environment. Will not work in standard Python scripts.

    Args:
        audio_path: Path to audio file
        annotation: 2D array with timestamps and beat positions
    """
    try:
        from IPython.display import Audio, display
    except ImportError:
        raise RuntimeError("play() requires IPython (use in Jupyter notebook)")
    
    y, sr = librosa.load(audio_path, sr=None)
    
    downbeats = annotation[annotation[:, 1] == 1, 0]
    other_beats = annotation[annotation[:, 1] != 1, 0]
    
    downbeat_clicks = librosa.clicks(times=downbeats, sr=sr, length=len(y), click_freq=1000)
    other_clicks = librosa.clicks(times=other_beats, sr=sr, length=len(y), click_freq=800)
    
    display(Audio(y + downbeat_clicks + other_clicks, rate=sr))


def evaluate(beats, downbeats, target: str | np.ndarray) -> tuple[float]:
    if isinstance(target, str):
        target = load(target)
    gt_beats = target[:, 0]
    gt_downbeats = target[target[:, 1] == 1, 0]
    
    beats_fscore = mir_eval.beat.f_measure(
        mir_eval.beat.trim_beats(gt_beats),
        mir_eval.beat.trim_beats(beats),
        0.07
    )
    
    downbeats_fscore = mir_eval.beat.f_measure(
        mir_eval.beat.trim_beats(gt_downbeats),
        mir_eval.beat.trim_beats(downbeats),
        0.07
    )
    
    return beats_fscore, downbeats_fscore