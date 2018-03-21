# Experiments with Meta-Learning
## Searching for beautiful minima

This project uses python 3.5 with Keras/tensorflow for experiments.<br>
Code works only for Keras with tensorflow backend.

To run experiments:
1. create conda environment or install required packages:<br>
```conda env create -f config/environment.yml``` <br>
or <br>
```pip install -r requirements.txt```<br>
**Note**: if GPU is not used, change 'tensorflow-gpu' to 'tensorflow'
2. set hyperparameters. As of now they need to be set by manually editing file `train.py`
(this will be refactored)
3. run: `python train.py`

For visualizations, see: `notebooks` folder
    
