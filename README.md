# Alpacapella: Beat Detection for Rap Acapellas
Beat tracking system for rap vocal tracks using deep learning.

## Setup
For the installation we first need a working distribution of python. This package can be installed using:
```bash
!pip install git+https://github.com/timowendner/alpacapella
```
Then the whole functionality can be imported:
```py
import alpacapella
```
A full example is given on [Google Colab](https://colab.research.google.com/drive/1vNaLYk-uEV-dVjRfm2-RoOEnQaiUX4yG#scrollTo=PVQx7hkE50KD).

## Functionality
### annotations module
The annotations module contains several useful functions for the handling of annotations, as well as the construction of annotations. The functionality includes:
- `alpacapella.annotations.pipeline`: Merge multiple annotations into single output using fill, smooth, vote, and downbeat prediction.
- `alpacapella.annotations.write_sample`: Save audio and annotation files to dataset directory.
- `alpacapella.annotations.create_dataset`: Process all annotations and create dataset with audio files.
- `alpacapella.annotations.play`: Play audio with beat clicks in Jupyter notebook.
- `alpacapella.annotations.load`: Load beat timestamps and downbeat positions from file.
- `alpacapella.annotations.save`: Save beat timestamps and downbeat positions to file.
- `alpacapella.annotations.evaluate`: Compute F-scores for beat and downbeat predictions.
- `alpacapella.annotations.statistics`: Collect beat counts, BPMs, and audio lengths from all files in folder.

### beat_this module
Using the [beat_this](https://github.com/CPJKU/beat_this) Repository to get predictions. The functionality includes:
- `alpacapella.beat_this.evaluate`: Get the F1-score of beats and downbeats, for a pre-trained beat_this model.

### madmom module
Using the [madmom](https://github.com/CPJKU/madmom) Repository to get predictions. The functionality includes:
- `alpacapella.madmom.evaluate`: Get the F1-score of beats and downbeats, for a pre-trained madmom model.

## Usage
### annotations module
**pipeline**: Merge multiple annotations into single output. Takes `annotation_path` (directory with annotation files), `smoothing_size` (window size for smoothing, default 1.5), `voting_window` (agreement threshold in seconds, default 0.07), `lag` (time offset, default 0.0), `is_plot` (whether to display plot, default True). Returns processed annotation array and percentage of real annotations.
```py
annotation, real_percentage = alpacapella.annotations.pipeline(
    'annotations/', smoothing_size=1.5, voting_window=0.07
)
```

**write_sample**: Save audio and annotation to dataset folder. Takes `audio_path` (source audio file), `dataset_path` (output directory), `annotation` (2D array with timestamps and beat positions), `file_name` (output name without extension), `cutoff` (seconds after last beat to keep, default 2.0).
```py
alpacapella.annotations.write_sample(
    'song.wav', 'dataset/', annotation, 'track001', cutoff=2.0
)
```

**create_dataset**: Process all annotations and create dataset. Takes `dataset_path` (output directory), `annotation_path` (input directory with subfolders), `smoothing_size` (default 2.2), `voting_window` (default 0.05), `cutoff` (default 2.0), `threshold` (minimum real annotation percentage, default 0.4).
```py
alpacapella.annotations.create_dataset(
    'dataset/', 'raw_annotations/', threshold=0.4
)
```

**play**: Play audio with click track overlay. Takes `audio_path` (path to audio file), `annotation` (2D array with timestamps and beat positions). Works only in Jupyter notebooks.
```py
alpacapella.annotations.play('song.wav', annotation)
```

**load**: Load beat annotations from file. Takes `annotation_path` (path to .beats or .txt file). Returns timestamps and beat positions as numpy array.
```py
annotation = alpacapella.annotations.load('path/to/file.beats')
```

**save**: Save beat annotations to file. Takes `annotation` (2D array with timestamps and beat positions), `annotation_path` (output file path).
```py
alpacapella.annotations.save(annotation, 'output.beats')
```

**evaluate**: Calculate F1-scores for beat predictions. Takes `beats` (predicted beat timestamps), `downbeats` (predicted downbeat timestamps), `target` (ground truth file path or array). Returns tuple of (beats_fscore, downbeats_fscore).
```py
beat_score, downbeat_score = alpacapella.annotations.evaluate(
    pred_beats, pred_downbeats, 'ground_truth.beats'
)
```

**statistics**: Collect beat counts, BPMs, and audio lengths. Takes `folder_path` (directory to search recursively). Returns lists of beat counts, BPMs, and audio durations.
```py
beats, bpms, lengths = alpacapella.annotations.statistics('dataset/')
```