# Flower Classification with Deep Learning
Training a convolutional neural network to recognize flowers on a prepared set of Flower17 graphics.

## Description:
The project demonstrates the process of building a convolutional neural network (CNN) to classify flower images into 17 different categories using TensorFlow and Keras.

The collection of photos used in this project is Flower17, you can download it here: http://www.robots.ox.ac.uk/~vgg/data/flowers/17/

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Code Structure](#code-structure)
3. [Stages](#stages)
4. [Results](#results)
5. [Testing the Model](#testing)

<a name="prerequisites"></a>
### 1. Prerequisites
To run this project, you need the following prerequisites:

* Python 3.x
* TensorFlow
* NumPy
* Matplotlib
* Scikit-learn

<a name="code-structure"></a>
### 2. Code Structure

The project code and files are organized as follows:

* __17flowers directory__: contains the "17 Category Flower Dataset"
* __flower-classification-deep-learning.py__: The main Python script and the core of the project
* __flowers17_model.h5__: This is the saved trained model
* __test.py__: A Python script used for testing the trained model with custom images
* __flower1.jpg__ and __flower2.jpg__: Sample custom images for testing the trained model

<a name="stages"></a>
### 3. Stages

__1. Data Preparation__
  * Loading images
  * Resizing images to 64x64 pixels
  * Storing images in numpy arrays.
  * Assigning class labels to the images.

__2. Data Splitting__

Splitting the dataset into training and testing sets (80/20).


__3. Data Preprocessing__

Normalizing pixel values to the range [0, 1].


__4. Data Augmentation__

Expanding the training dataset through augmentation techniques, such as rotation, shifts, shear, zoom, and horizontal flips, utilizing the ImageDataGenerator class.


__5. Model Architecture__

Defining a CNN model with convolutional layers, max-pooling layers, and fully connected layers. Dropout is employed for overfitting prevention.


__6. Model Compilation__

Implementing learning rate scheduling.
Compiling the model with the Adam optimizer and sparse categorical cross-entropy loss.


__7. Model Training__

Training the model for 100 epochs with data augmentation for enhanced robustness.


__8. Model Evaluation__

Evaluating model performance on the testing dataset, including metrics like test loss, accuracy, and a classification report with precision, recall, and F1-score for each class.


__9. Visualization__

Generating accuracy and loss plots to visualize training and validation performance.

__10. Saving the Model__

Saving the trained model as "flowers17_model.h5" for future use.

<a name="results"></a>
### 4. Results

The trained model achieved the following results on the test set:

Test Loss: __1.0289__ </br>
Test Accuracy: __0.7610__

Training and Validation Loss:

![alt text](https://github.com/abrylanska/flower-classification-deep-learning/blob/master/traning_validation_loss.JPG?raw=true)

Training and Validation Accuracy:

![alt text](https://github.com/abrylanska/flower-classification-deep-learning/blob/master/training_validation_acc.JPG?raw=true)

Metrics:

![alt text](https://github.com/abrylanska/flower-classification-deep-learning/blob/master/additional_results.JPG?raw=true)


<a name="testing"></a>
### 5. Testing the Model

To demonstrate the use of the trained model to predict the class of a new image, it was tested on two photos of flowers downloaded from the Internet.

The first flower is a pansy:

![pansy](https://github.com/abrylanska/flower-classification-deep-learning/blob/master/flower1.jpg?raw=true)

The second flower is a daisy:

![daisy](https://github.com/abrylanska/flower-classification-deep-learning/blob/master/flower2.jpg?raw=true)

__The results:__

In the case of the pansy, the model classified it in the correct class (16).
Previous metrics for the pansy class were as follows:
* precision: 0.95
* recall: 0.90
* f1-score: 0.92
* support: 20

```
1/1 [==============================] - 0s 432ms/step
Predicted class for the image: 16
```

In the case of the daisy, the model also classified it into the correct class (10).
The previous metrics for the daisy class were as follows:
* precision: 0.89
* recall: 1.0
* f1-score: 0.94
* support: 16

```
1/1 [==============================] - 0s 128ms/step
Predicted class for the image: 10
```

It is worth mentioning that a small set of photos was selected to train the model. For better recognition, the number of samples should be larger.
