from beat_this.inference import File2Beats
import numpy as np
from .. import annotations

def evaluate(audio_path: str, annotation: str | np.ndarray):
    file2beats = File2Beats(checkpoint_path="final0", device="cpu", dbn=False)
    beats, downbeats = file2beats(audio_path)

    beats_fscore, downbeats_fscore = annotations.evaluate(
        beats, downbeats, annotation
    )
    return beats_fscore, downbeats_fscore