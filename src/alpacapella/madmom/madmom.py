import numpy as np
from madmom.audio.signal import Signal
from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
from .. import annotations

def predict(audio: str | np.ndarray, sr: int = 44100):
    """Get beat and downbeat predictions based on audio.

    Args:
        audio: Path to audio file or loaded array of shape (N, 2)
        sr: The sample rate of the audio file (Only important for loaded arrays)
    
    Returns:
        tuple of predictions for beats and downbeats.
    """
    if not isinstance(audio, str) and sr != 44100:
        audio = Signal(audio, sample_rate=sr)

    rnn_processor = RNNDownBeatProcessor()
    dbn_processor = DBNDownBeatTrackingProcessor(beats_per_bar=[4], min_bpm=60, max_bpm=200, fps=100)

    activations = rnn_processor(audio)
    beat_downbeat_positions = dbn_processor(activations)

    beats = beat_downbeat_positions[:, 0]
    downbeats_mask = beat_downbeat_positions[:, 1] == 1
    downbeats = beat_downbeat_positions[downbeats_mask, 0]

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

    beats_score, downbeats_score = annotations.evaluate(
        beats, downbeats, annotation
    )
    
    return beats_score, downbeats_score