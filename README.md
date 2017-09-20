# convolutional_handshape
Repository of code for the experiments in "A Study of Convolutional Architectures for Handshape Recognition applied to Sign Language"

There are two sets of experiments:
* Transfer learning experiments, in folder `transfer_experiments`
* _Normal_ experiments (training a net end-to-end), in folder `normal_experiments`

## Normal experiments
To run the normal experiments, install the packages listed in `normal_experiments/code/requirements.txt` and either `normal_experiments/code/requirements-tf.txt` or `normal_experiments/code/requirements-tf-gpu.txt`.

Then run `python experiment.py` from the folder `normal_experiments/code/`. You edit the file `experiments.py` to  select the dataset and model, as well as the parameters.

## Transfer experiments


