from beat_this.inference import File2Beats, Audio2Beats
import numpy as np
from .. import annotations

def predict(audio: str | np.ndarray, sr=None, device: str = "cpu", dbn: bool = False):
    """Get beat and downbeat predictions based on audio.

    Args:
        audio: Path to audio file or loaded array of shape (N, 2)
        sr: The sample rate of the audio file (Only important for loaded arrays)
        dbn: Whether the dynamic bayesian network is used.
    
    Returns:
        tuple of predictions for beats and downbeats.
    """
    checkpoint_path = "final0"
    if isinstance(audio, np.ndarray):
        model = Audio2Beats(checkpoint_path, device, dbn=dbn)
        beats, downbeats = model(audio, sr)
    else:
        model = File2Beats(checkpoint_path, device, dbn=dbn)
        beats, downbeats = model(audio)
    return beats, downbeats


def evaluate(audio: str | np.ndarray, annotation: str | np.ndarray):
    """Evaluate beat and downbeat predictions against ground truth annotations.

    Args:
        audio: Path to audio file or loaded array of shape (N, 2)
        target: Path to annotation file or loaded annotation array with shape (N, 2)

    Returns:
        tuple of beat and downbeat metric (f1, cmlt, amlt).
    """
    beats, downbeats = predict(audio)

    beats_fscore, downbeats_fscore = annotations.evaluate(
        beats, downbeats, annotation
    )
    return beats_fscore, downbeats_fscore