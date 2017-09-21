## convolutional_handshape
Repository of code for the experiments in "A Study of Convolutional Architectures for Handshape Recognition applied to Sign Language"

There are two sets of experiments:
* Transfer learning experiments, in folder `transfer_experiments`
* _Normal_ experiments (training a net end-to-end), in folder `normal_experiments`

## Normal experiments
To run the normal experiments, install the packages listed in `normal_experiments/code/requirements.txt` and either `normal_experiments/code/requirements-tf.txt` or `normal_experiments/code/requirements-tf-gpu.txt`.

Then run `python experiment.py` from the folder `normal_experiments/code/`. You edit the file `experiments.py` to  select the dataset and model, as well as the parameters.

## Transfer experiments
This experiment has the same requirements as the normal experiments. To run the transfer experiment we have to follow three steps:
* First, run `python create_tmp_dirs.py` to generate the JPG labeled files to feed the Inception network.
* Then the feature extraction from Inception is made running `python feature_extraction.py -i [IMAGE_DIR]`. This will create three files containing the features, the labels and a JSON with human readable labels.
* Last, run `python train_svm -f [FEATURES_PKL] -l [LABELS_PKL] -j [LABELS_JSON]` or `python train_nn.py -f [FEATURES_PKL] -l [LABELS_PKL] -j [LABELS_JSON]` to train either a SVM or a feedforward Neural Network with the features extracted.


