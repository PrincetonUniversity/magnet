[![Build](https://github.com/PrincetonUniversity/magnet/actions/workflows/main.yml/badge.svg)](https://github.com/PrincetonUniversity/magnet/actions/workflows/main.yml)

## Tutorial

This is a brief tutorial of how to train and test a transformer-based encoder-projector-decoder neural network model for the hysteresis loop prediction of power magnetics.

The training part can be found in `Demo_Transformer_Train.ipynb`, where a simple network model is trained for the sinusoidal waveform predictions. The model is trained based on the dataset `..\Dataset\Dataset_sine.json`. The trained model is saved as state dictionary file `.\Output\Model_Transformer.sd`. 

The testing part can be found in `Demo_Transformer_Test.ipynb`, and the testing results and their references are saved in `.\Output\pred.csv` and `.\Output\meas.csv`.

To simplify the learning process and get a quick start, we suggest downloading the notebook file and executing it step-by-step in the [Google Golab](https://colab.research.google.com/) (free).