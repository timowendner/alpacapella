from beat_this.inference import File2Beats
import numpy as np
from .. import annotations

def predict(audio_path: str):
    file2beats = File2Beats(checkpoint_path="final0", device="cpu", dbn=False)
    beats, downbeats = file2beats(audio_path)
    return beats, downbeats

def evaluate(audio_path: str, annotation: str | np.ndarray):
    beats, downbeats = predict(audio_path)

    beats_fscore, downbeats_fscore = annotations.evaluate(
        beats, downbeats, annotation
    )
    return beats_fscore, downbeats_fscore