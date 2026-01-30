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


