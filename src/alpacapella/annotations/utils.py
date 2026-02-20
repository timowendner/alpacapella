import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import mir_eval

def load(annotation_path: str) -> np.ndarray:
    """Load beat timestamps and downbeat positions from file.

    Args:
        annotation_path: Path to annotation file
    
    Returns:
        Beat timestamp and downbeat array
    """
    annotation = np.loadtxt(annotation_path)
    return annotation

def save(annotation: np.ndarray, annotation_path: str):
    """Save beat timestamps and downbeat positions to file.

    Args:
        annotation: Beat timestamp and downbeat array
        annotation_path: Output file path
    """
    np.savetxt(annotation_path, annotation, fmt=['%.9f', '%d'])
    


def load_folder(annotation_path: str) -> tuple[np.ndarray]:
    """Load all .txt annotation files from directory.

    Args:
        annotation_path: Directory containing annotation files
    
    Returns:
        List of beat and measure timestamp arrays
    """
    annotations = []
    measure = []
    for file in sorted(os.listdir(annotation_path)):
        if not file.endswith('.txt') or file.endswith('.beats'):
            continue
        file_path = os.path.join(annotation_path, file)
        annotation = load(file_path)
        annotations.append(np.sort(annotation[:, 0]))
        measure.append(annotation[:, 1])
    return annotations, measure


def plot(raw: np.ndarray, annotation: np.ndarray, title: str, window_ms: int = 70):
    """Plot raw vs final beat annotations with time windows.

    Args:
        raw: Combined timestamps from all raw annotations
        annotation: 2D array with timestamps and beat positions
        title: Plot title
        window_ms: Window size in milliseconds for visualization
    """
    window_s = window_ms / 1000.0
    half_window = window_s / 2.0
    fig, ax = plt.subplots(figsize=(12, 3))
    
    for i, beat in enumerate(annotation[:, 0]):
        if i == 0:
            ax.axvspan(beat - half_window, beat + half_window, color='gray', alpha=0.3, label=f'{window_ms}ms Window')
        else:
            ax.axvspan(beat - half_window, beat + half_window, color='gray', alpha=0.3)
            
    ax.vlines(raw, 0.1, 0.45, colors='blue', label='Raw')
    ax.vlines(annotation[:, 0], 0.55, 0.9, colors='orange', label='Final')
    
    ax.set_yticks([1, 2])
    ax.set_yticklabels(['Raw', 'Final'])
    ax.set_xlabel('Time (seconds)')
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.show()

def play(audio_path: str, annotation: np.ndarray):
    """Play audio with beat clicks in Jupyter notebook.

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


def evaluate(beats, downbeats, target: str | np.ndarray) -> tuple[dict[str, float], dict[str, float]]:
    """Evaluate beat and downbeat predictions against ground truth annotations.

    Args:
        beats: Predicted beat timestamps
        downbeats: Predicted downbeat timestamps
        target: Path to annotation file or loaded annotation array with shape (N, 2)

    Returns:
        tuple of beat and downbeat metric (f1, cmlt, amlt).
    """
    if isinstance(target, str):
        target = load(target)
    
    gt_beats = target[:, 0]
    gt_downbeats = target[target[:, 1] == 1, 0]
    
    beats = mir_eval.beat.trim_beats(beats)
    gt_beats = mir_eval.beat.trim_beats(gt_beats)

    f1score = mir_eval.beat.f_measure(gt_beats, beats, 0.07)
    CMLc, CMLt, AMLc, AMLt = mir_eval.beat.continuity(gt_beats, beats)

    beats_metrics = {
        "f1": f1score,
        "cmlt": CMLt,
        "amlt": AMLt
    }

    gt_downbeats = mir_eval.beat.trim_beats(gt_downbeats)
    downbeats = mir_eval.beat.trim_beats(downbeats)
    
    f1score = mir_eval.beat.f_measure(gt_downbeats, downbeats, 0.07)
    CMLc, CMLt, AMLc, AMLt = mir_eval.beat.continuity(gt_downbeats, downbeats)
    
    
    downbeats_metrics = {
        "f1": f1score,
        "cmlt": CMLt,
        "amlt": AMLt
    }
    
    return beats_metrics, downbeats_metrics


def remove_silence(
        audio: str | np.ndarray,
        annotation: str | np.ndarray,
        hop_size: int = 441,
        dilation_frames: int = 100,
        erosion_frames: int = 25,
        sr: int = 44100,
    ):
    if isinstance(audio, str):
        audio, orig_sr = librosa.load(audio, sr=None)
        if orig_sr != sr:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)
    if isinstance(annotation, str):
        annotation = load(annotation)

    rms = librosa.feature.rms(y=audio, frame_length=hop_size, hop_length=hop_size)[0]
    threshold = np.percentile(rms[rms > 0], 20)
    loud = (rms > threshold)

    dilation_frames = 2*dilation_frames + 1
    erosion_frames = 2*erosion_frames + 1
    kernel_d = np.ones(dilation_frames)
    kernel_e = np.ones(erosion_frames)

    dilated = (np.convolve(loud, kernel_d, mode='full') > 0)
    eroded = (np.convolve(dilated, kernel_e, mode='same') >= erosion_frames)
    eroded = eroded[dilation_frames // 2: -(dilation_frames // 2)]

    result = []
    for time, measure in annotation:
        sample = int(time * sr / hop_size)
        if not (0 <= sample < len(loud)):
            continue
        if not eroded[sample]:
            continue
        result.append([time, measure])
    return np.array(result)