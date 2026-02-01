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
- `alpacepella.annotations.play`: Play the audio with an annotation click track.
- `alpacepella.annotations.load`: Load the annotation and get the correct format.
- `alpacepella.annotations.evaluate`: Get the F1-score of beats and downbeats.
- `alpacepella.annotations.pipeline`: Use multiple annotations to create a combination of those annotations.
- `alpacepella.annotations.write_dataset`: Write the annotation as well as the audio to the dataset folder.

### beat_this module
Using the [beat_this](https://github.com/CPJKU/beat_this) Repository to get predictions. The functionality includes:
- `alpacapella.beat_this.evaluate`: Get the F1-score of beats and downbeats, for a pre-trained beat_this model.

### madmom module
Using the [madmom](https://github.com/CPJKU/madmom) Repository to get predictions. The functionality includes:
- `alpacapella.madmom.evaluate`: Get the F1-score of beats and downbeats, for a pre-trained madmom model.

## Usage
### annotations module
**load**: Load beat annotations from file. Takes `annotation_path` (path to .beats or .txt file). Returns timestamps and beat positions as numpy array.
```py
annotation = alpacapella.annotations.load('path/to/file.beats')
```

**play**: Play audio with click track overlay. Takes `audio_path` (path to audio file), `annotation` (2D array with timestamps and beat positions). Works only in Jupyter notebooks.
```py
alpacapella.annotations.play('song.wav', annotation)
```

**evaluate**: Calculate F1-scores for beat predictions. Takes `beats` (predicted beat timestamps), `downbeats` (predicted downbeat timestamps), `target` (ground truth file path or array). Returns tuple of (beats_fscore, downbeats_fscore).
```py
beat_score, downbeat_score = alpacapella.annotations.evaluate(
    pred_beats, pred_downbeats, 'ground_truth.beats'
)
```

**pipeline**: Merge multiple annotations into single output. Takes `annotation_path` (directory with annotation files), `smoothing_size` (window size for smoothing, default 2.2), `voting_window` (agreement threshold in seconds, default 0.05). Returns processed annotation array and percentage of real annotations.
```py
annotation, real_percentage = alpacapella.annotations.pipeline(
    'annotations/', smoothing_size=2.2, voting_window=0.05
)
```

**write_dataset**: Save audio and annotation to dataset folder. Takes `audio_path` (source audio file), `dataset_path` (output directory), `annotation` (2D array with timestamps and beat positions), `file_name` (output name without extension), `cutoff` (seconds after last beat to keep, default 2.0).
```py
alpacapella.annotations.write_dataset(
    'song.wav', 'dataset/', annotation, 'track001', cutoff=2.0
)
```

**create_dataset**: Process multiple annotation folders into complete dataset. Takes `dataset_path` (output directory), `annotation_path` (input directory with subfolders), `smoothing_size` (default 2.2), `voting_window` (default 0.05), `cutoff` (default 2.0), `threshold` (minimum real annotation percentage, default 0.4). Creates numbered files in output directory.
```py
alpacapella.annotations.create_dataset(
    'dataset/', 'raw_annotations/', threshold=0.4
)
```