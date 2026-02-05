from beat_this.inference import File2Beats, Audio2Beats
import numpy as np
from .. import annotations

def predict(audio: str | np.ndarray, sr=None, device: str = "cpu"):
    checkpoint_path = "final0"
    if isinstance(audio, np.ndarray):
        model = Audio2Beats(checkpoint_path, device, dbn=False)
        beats, downbeats = model(audio, sr)
    else:
        model = File2Beats(checkpoint_path, device, dbn=False)
        beats, downbeats = model(audio)
    return beats, downbeats


def evaluate(audio: str | np.ndarray, annotation: str | np.ndarray):
    beats, downbeats = predict(audio)

    beats_fscore, downbeats_fscore = annotations.evaluate(
        beats, downbeats, annotation
    )
    return beats_fscore, downbeats_fscore