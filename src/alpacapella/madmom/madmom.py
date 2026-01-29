import numpy as np
from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
from .. import annotations

def evaluate(audio_file, annotation: str | np.ndarray):
    rnn_processor = RNNDownBeatProcessor()
    dbn_processor = DBNDownBeatTrackingProcessor(beats_per_bar=[4, 8], fps=100)
    
    activations = rnn_processor(audio_file)
    beat_downbeat_positions = dbn_processor(activations)
    
    beats = beat_downbeat_positions[:, 0]
    downbeats_mask = beat_downbeat_positions[:, 1] == 1
    downbeats = beat_downbeat_positions[downbeats_mask, 0]

    beats_fscore, downbeats_fscore = annotations.evaluate(
        beats, downbeats, annotation
    )
    
    return beats_fscore, downbeats_fscore