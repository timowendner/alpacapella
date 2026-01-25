# AlpAcapella: Beat Detection for Rap Acapellas
Beat tracking system for rap vocal tracks using deep learning.

## Setup
For the installation we first need a working distribution of python. This package can be installed using:
```python
!pip install git+https://github.com/timowendner/alpacapella
```
A full example is given on [Google Colab](https://colab.research.google.com/drive/1vNaLYk-uEV-dVjRfm2-RoOEnQaiUX4yG#scrollTo=PVQx7hkE50KD).

We can now use the Annotation Pipeline
```python
import alpacapella

ANNOTATION_PATH = "notebooks/examples"
AUDIO_PATH = "notebooks/examples/example_audio.wav"

annotation = alpacapella.pipeline(
    ANNOTATION_PATH, smoothing_size=2.2, voting_window=0.05
)
alpacapella.play(AUDIO_PATH, annotation)
```

## Datasets
- Annotation Dataset (AD): manually annotated Looperman acapellas

