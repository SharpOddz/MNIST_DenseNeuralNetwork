# MNIST_DenseNeuralNetwork
Utilizing a Fully Connected Neural Network to perform handwritten digit classification on MNIST dataset. The code is designed to make it easy to adjust the model's hyperparameters, architecture, and training process. 

## Requirements
The code automatically downloads the MNIST dataset at the start of program execution so a internet connection is required to run it. Required libraries: tensorflow, sklearn, matplotlib

## Preprocessing:
Pixel values are normalized and the image is flattened (28x28)->(784x1). A training/validation split of 90/10 is used for this model.

## Model Hyperparameters and Architecture
The model hyperparameters can be easily adjusted, however the best results (99.86% training set accuracy, 98.82% validation accuracy on 90/10 train/val split) are with the following parameters:
 - Epochs: 20
 - Learning rate: 0.001 with reduction by factor of 0.2 if 3 epochs without validation loss (minimum learning rate of 0.00001)
 - Dropout rate: 0.2
 - Batch normalization
 - Relu activations
 - L1 and L2 regularization turned off
 - Model Architecture: (784->512->256-128->64->10)
 - Loss: sparse categorical cross entropy
 - Optimizer: Adam

The model calls back to its best version if the last epoch run is not the best in terms of val loss. 

## Output
The model outputs the following:
  - Training, validation, and testing data shapes
  - Model architecture/summary
  - Training progress bars for each epoch
  - Best training and validation accuracy/loss
  - Plots for training and validation loss and accuracy over the training process (red dotted line signifies the best model run)

