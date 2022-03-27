# audio-style-transfer

### Daniel Quintao de Moraes, Yasmine Guemouria, Yousra Leouafi, Oumaima Marbouh

This project was realized for an "Applied Deep learning with Python" class by M2 Data science students at IP Paris.

### Goal
Implement audio style transfer with Pytorch, steps inspired by "AUDIO STYLE TRANSFER WITH RHYTHMIC CONSTRAINTS, Maciek Tomczak, Carl Southall and Jason Hockman". We foccus only on the combination where one audio gives s the content and the other one gives us the style (Tomczak et al tried other combinations as well).

### Our work
First of all, we decided to use PyTorch since the paper's provided code and most of the public repos and notebooks that do AST (Audio Style Transfer) in this more "traditional" manner (i.e. inspired from Style Transfer in images) use Tensorflow (or Theano). In the case of [Tomczak et al](https://github.com/maciek-tomczak/audio-style-transfer-with-rhythmic-constraints) it is aso compatible with Python 2 only. Other works that helped ours were [this repo by alishdipani](https://github.com/alishdipani/Neural-Style-Transfer-Audio) and mostly [this repo by Lebedev and Ulyanov](https://github.com/DmitryUlyanov/neural-style-audio-tf).

Although the architecture in Tomczak et al is very simple, doing the optimization showed really challenging. Moreover, we could not manage to use PyTorch's [LBFGS](https://pytorch.org/docs/stable/generated/torch.optim.LBFGS.html) (we had some bugs that seem to be some issue of the framework according to some research on forums). We finally used AdamW optimizer of PyTorch which is not bad, combined with a learning-rate scheduler and some code to do Early Stopping that we added because the content file was dominating the style file in our first essays.

### Data set
We did our tests with audio tracks used by [Tomczak et al](https://maciek-tomczak.github.io/maciek.github.io/Audio-Style-Transfer-with-Rhythmic-Constraints/) and some audio files used by [Grinstein et al](https://egrinstein.github.io/2017/10/25/ast.html).

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
  
