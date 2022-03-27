# audio-style-transfer

This project was realized for an "Applied Deel learning with Python" class by M2 Data science students at IP Paris.

### Goal
Implement audio style transfer with Pytorch, steps inspired by "AUDIO STYLE TRANSFER WITH RHYTHMIC CONSTRAINTS, Maciek Tomczak, Carl Southall and Jason Hockman".

### Project structure walkthrough
- data folder: contains audio files that we will use as content and style inputs for our model
- outputs: contains spectrogram magnitudes (log-scale) in .npy format and inside it resides a subfolder "audio_results" where the produced mixed audio by the model is exported
- src:
  - config: 
    - vars.yaml: a yaml file that contains all global variables used in the project
  - optimize_truncated.py: model and training
  - test.py: testing model with custom inputs and visualizations

### Execution instructions
as simple as:
1. run optimize_truncated.py to train model
2. run test.py to generate mixed audio
  