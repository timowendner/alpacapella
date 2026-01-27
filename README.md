# AlpAcapella: Beat Detection for Rap Acapellas
Beat tracking system for rap vocal tracks using deep learning.

## Setup
For the installation we first need a working distribution of python. This package can be installed using:
```bash
!pip install git+https://github.com/timowendner/alpacapella
```
Then the whole functionality can be imported:
```python
import alpacapella
```
A full example is given on [Google Colab](https://colab.research.google.com/drive/1vNaLYk-uEV-dVjRfm2-RoOEnQaiUX4yG#scrollTo=PVQx7hkE50KD).

## Functionality
### Annotation Pipeline
With `alpacapella.pipeline` the complete Pipeline is applied. The `annotation_path` interprets every textfile in this folder as annotations, and combines them. The `smoothing_size` specifies the window size for the smoothing operation. The `voting_window` is the biggest distance where all annotations have to agree, in order for this beat to count.
```py
annotation_path = "examples"

annotation = alpacapella.pipeline(
    annotation_path, smoothing_size=2.2, voting_window=0.05
)
```
To play the audio with the annotation `alpacapella.play` can be used. Arguments must include the `audio_path`, as well as the calculated `annotation`.
```py
audio_path = "examples/example_audio.wav"
alpacapella.play(audio_path, annotation)
```
If the annotations are correct, they can be saved with `alpacapella.write_dataset`. The `audio_path`, `dataset_path` and `annotation` must be provided. With `beats_in_bar` it can be specified how many beats are in a bar (most common for rap is 4 or 8). The `cutoff` is the time in seconds after the last annotation, where the audio is cut off.
```py
dataset_path = "examples/dataset"
alpacapella.write_dataset(
    audio_path, dataset_path, annotation, beats_in_bar=4, cutoff=2
)
```

