# Introduction
In this notebook, I have created and trained a Convolutional Neural Network (CNN) model to classify images from the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The classes are: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

# Dataset
I have used the Keras library to download the CIFAR-10 dataset. We have divided the dataset into training, validation, and testing sets. We have also preprocessed the data by scaling the pixel values to be between 0 and 1 and one-hot encoding the target labels.

<img src="https://github.com/SanthoshV14/cifar10-image-prediction-cnn.ipynb/blob/main/img/dataset.png" />

# Model Architecture
Created a CNN model using Keras. The model consists of two convolutional layers with 64 and 128 filters respectively, followed by max pooling layers. The output from the convolutional layers is flattened and passed through a dropout layer with a rate of 0.5 to prevent overfitting. Finally, a fully connected layer with 10 units and a softmax activation function is used to make predictions.

# Training the Model
The model was compiled using mean squared error loss and the Adam optimizer. Trained the model for 30 epochs using a batch size of 32. Evaluated the model on the validation set during training to monitor its performance.

# Results
The model has less than 100K learnable parameters and achieved an accuracy of 75.14% on the validation set.

<img src="https://github.com/SanthoshV14/cifar10-image-prediction-cnn.ipynb/blob/main/img/accuracy-plot.png" />

# Conclusion
In this notebook, I have demonstrated how to create and train a CNN model using Keras on the CIFAR-10 dataset. The model achieved good accuracy on the validation set and have shown that a simpler model can also achieve decent accuracy. This notebook can serve as a starting point for further experimentation with CNNs on image classification tasks.
