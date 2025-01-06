# Churn-Prediction-using-ANN

# Churn Prediction Model

This project aims to predict customer churn using a deep learning model built with **Keras** and **TensorFlow**. By analyzing customer data, the model helps businesses identify customers who are likely to leave, so that retention strategies can be implemented.

## Table of Contents

- [Project Overview](#project-overview)
- [Getting Started](#getting-started)
- [Requirements](#requirements)
- [Data Description](#data-description)
- [Model Architecture](#model-architecture)
- [Model Training](#model-training)
- [Evaluating the Model](#evaluating-the-model)
- [Visualization](#visualization)
- [License](#license)

## Project Overview

This is a **binary classification** problem where the goal is to predict whether a customer will churn (leave) or not based on their profile and other relevant features. The model is built using a **neural network** with **Keras** and trained on customer data.

## Getting Started

### Prerequisites

To run this project locally, make sure you have the following libraries installed:

- **Python 3.x**
- **Pandas**
- **Numpy**
- **Matplotlib**
- **Keras**
- **TensorFlow**
- **Scikit-learn**

You can install the required libraries using `pip`:


pip install numpy pandas matplotlib scikit-learn keras tensorflow
Running the Code
Clone this repository to your local machine:


Copy code
git clone https://github.com/your-username/churn-prediction.git
cd churn-prediction
Run the ChurnPredictionModel.py script:


Copy code
python ChurnPredictionModel.py
This will train the model, plot the accuracy and loss graphs, and output the predictions and confusion matrix.

Requirements
Python 3.x: Make sure Python 3.x is installed on your machine.
Pandas: For data manipulation and preprocessing.
Numpy: For numerical calculations.
Matplotlib: For plotting the training history.
Scikit-learn: For model evaluation and splitting data.
Keras and TensorFlow: For building and training the neural network model.
Data Description
The dataset used for this project is the Churn_Modelling.csv file, which contains customer information including:

Geography: The country of the customer (France, Germany, Spain).
Gender: The gender of the customer (Male, Female).
Age, Tenure, Balance, etc.: Various customer attributes like age, tenure, balance, etc.
The target variable is Exited, where:

1 indicates the customer has churned (left the company).
0 indicates the customer has not churned (stayed with the company).
Model Architecture
The neural network used in this project consists of:

Input Layer: 11 features (after encoding categorical variables).
Hidden Layers: 3 layers, each with 6 neurons, using ReLU activation function.
Output Layer: 1 neuron with a sigmoid activation function for binary classification.
Code Explanation:
Data Preprocessing:

One-Hot Encoding: Converts categorical variables (like Geography and Gender) into binary values.
Feature Scaling: Scales the features to have a mean of 0 and a standard deviation of 1, which helps the model converge faster.
Model Training:

The model is trained using the Adam optimizer and binary cross-entropy loss function. Accuracy is used as the metric.
The model is trained for 100 epochs with a batch size of 10 and a validation split of 0.33.
Model Evaluation:

Predictions are made on the test set, and the results are evaluated using accuracy score and a confusion matrix.
Visualizations are provided for both accuracy and loss during training.
Model Training
The model is trained using the following code:

python
Copy code
model_history = classifier.fit(X_train, y_train, validation_split=0.33, batch_size=10, epochs=100)
X_train and y_train: Training data and labels.
Validation Split: 33% of the data is used for validation.
Batch Size: 10 samples per batch.
Epochs: The model is trained for 100 epochs.
Evaluating the Model
Once the model is trained, the performance is evaluated on the test set:

Confusion Matrix: Shows how well the model is classifying the test data.
Accuracy Score: Provides the overall accuracy of the model.
Visualization
The accuracy and loss curves are plotted for both training and validation data to visualize the model's performance over the epochs.

python
Copy code
# Accuracy plot
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
