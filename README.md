# Caffe Plugin
## About
# Language: Python
# Input: CSV (unnormalized matrix)
# Output: CSV (matrix normalized across rows)
# Tested with: PluMA 1.0, Python 3.6, CUDA 8.0

This plugin allows one to input a neural network for classifcation as defined by two caffe files that define the weights and model architecture.

## What to pass in
1. prototxt file that descrbes model architecture
1. Caffemodel with pre-trained weights
1. Pickled data file Should already be preprocessed)
1. Tab delimited labels file

## Network constraints
* This plugin assumes that your classification network has a last layer which returns an array of probabilities, the index of the highest of these probabilities is then used as the index for the labels array. 
* This plugin utilizes the GPU.
