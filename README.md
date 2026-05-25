# MNIST_DenseNeuralNetwork
Utilizing a Fully Connected Neural Network to perform handwritten digit classification on MNIST dataset. Optional features include: dropout, L1 and/or L2 regularization. All hyperparemters, including optional features, can be adjusted easily. Train and validation split is currently set at 90/10 but can be changed. For preprocessing, pixel values are normalized. Model includes early stopping if 5 or more epochs without improvement to validation loss. The best model in terms of validation loss is saved if validation loss increases before stopping. Program outputs the train and validation accuracy and loss in two plots.

Written in Python and utilizing Tensorflow
