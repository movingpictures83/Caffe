# Caffe
# Language: Python
# Input: TXT
# Output: TXT
# Tested with: PluMA 1.1, Python 3.6
# Dependency: numpy==1.16.0, caffe==1.0.0

This plugin allows one to input a neural network for classifcation as defined by two caffe files that define the weights and model architecture.

## What to pass in
1. prototxt file that descrbes model architecture
1. Caffemodel with pre-trained weights
1. Pickled data file Should already be preprocessed)
1. Tab delimited labels file

## Network constraints
* This plugin assumes that your classification network has a last layer which returns an array of probabilities, the index of the highest of these probabilities is then used as the index for the labels array. 
* This plugin utilizes the GPU.
