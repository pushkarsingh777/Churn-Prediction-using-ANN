# Churn Prediction Model

This project predicts customer churn using a deep learning model built with **Keras** and **TensorFlow**. By analyzing customer data, businesses can identify which customers are likely to leave, allowing for targeted retention strategies.

## Requirements

To run this project, you need:

- Python 3.x
- Pandas
- Numpy
- Matplotlib
- Scikit-learn
- Keras
- TensorFlow

Install the necessary libraries:

```bash
pip install numpy pandas matplotlib scikit-learn keras tensorflow

Data Description
The dataset used is Churn_Modelling.csv with the following key features:

Geography: Country (France, Germany, Spain)
Gender: Customer gender (Male, Female)
Age, Tenure, Balance, etc.: Customer information
Exited: Target variable (1 for churn, 0 for no churn)
Model Architecture
Input Layer: 11 features after encoding.
Hidden Layers: 3 layers with 6 neurons and ReLU activation.
Output Layer: 1 neuron with sigmoid activation for binary classification.
Model Training
The model is trained using the Adam optimizer, binary cross-entropy loss function, and accuracy as the metric. It is trained for 100 epochs with a batch size of 10.
