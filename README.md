# MNIST Digit Classification using Feedforward Neural Network

This project implements a simple neural network using TensorFlow and Keras to classify handwritten digits from the MNIST dataset.

## Project Overview

The goal is to build a feedforward neural network (also called a dense or fully connected neural network) that takes flattened 28x28 pixel images (784 features) and classifies them into digits from 0 to 9.

This project helped me understand the basics of how neural networks work for image classification tasks.

## Tools & Technologies

- Python  
- TensorFlow & Keras  
- NumPy  
- Matplotlib  
- Jupyter Notebook  

## Dataset

The [MNIST dataset](http://yann.lecun.com/exdb/mnist/) is a benchmark dataset in computer vision, consisting of 70,000 grayscale images of handwritten digits (60,000 for training and 10,000 for testing).

Each image is 28x28 pixels, and the dataset is already available in Keras via `keras.datasets.mnist`.

## Model Architecture

- **Input Layer:** 784 nodes (flattened 28x28 image)  
- **Dense Layer:** 10 neurons (one for each digit), using the sigmoid activation function  

```python
model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(784,), activation='sigmoid')
])
```
- Loss Function: sparse_categorical_crossentropy
- Optimizer: adam
- Metrics: Accuracy

## Training

The model is trained for 5 epochs using the Adam optimizer. After training, it reaches an accuracy of around 97% on the training data.
```
model.fit(X_train_flattened, y_train, epochs=5)
```
## Results

    The model successfully classifies most digits from the test dataset.

    It's a basic but important step in understanding how machine learning models learn from visual data.

## Learning Outcome

This project was inspired by a YouTube tutorial and helped me build a foundational understanding of neural networks in computer vision. I now have a better grasp of model architecture, activation functions, and loss calculation.

How to Run

    Clone this repository

    Install required libraries: tensorflow, matplotlib, numpy

    Open the notebook: Number_Identification_Model.ipynb in Jupyter

    Run all cells


    Note: This project is for educational and learning purposes only. It serves as a foundation for understanding neural networks and image classification. For production use or more complex applications, consider advanced architectures and proper model validation
