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
Getting Started
To get started with this project, clone the repository to your local machine and navigate into the project directory:


git clone https://github.com/your-username/churn-prediction.git
cd churn-prediction
After running these commands, you will have the project files on your local machine. Now, you can set up the environment and install the necessary dependencies.

Setting Up
Make sure you have Python 3.x installed. Then, install the required dependencies by running:


pip install -r requirements.txt
This will install all the necessary libraries to run the project. Once the setup is complete, you can begin training the model or making predictions.

Model Training
To train the model, you can run the following command:


python ChurnPredictionModel.py
This will train the neural network on the provided dataset and output the training and validation accuracy over 100 epochs. You will also see visualizations for model performance.

Description
In this project, we use a Deep Learning model built with Keras to predict customer churn based on various customer attributes. The dataset is preprocessed to handle categorical variables and scaled before being fed into the neural network.

The model uses an Artificial Neural Network (ANN) with the following architecture:

Input Layer: 11 features after one-hot encoding categorical data
Hidden Layers: 3 hidden layers with ReLU activation functions
Output Layer: A single neuron with sigmoid activation for binary classification
By the end of training, the model will predict whether a customer is likely to churn (1) or not (0).

Visualization
Once the training is complete, you can visualize the accuracy and loss for both training and validation sets over time.


# Accuracy plot
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
License
This project is licensed under the MIT License.


### How to Use:
- Just copy and paste this code into a **README.md** file in your GitHub project.
- Replace `your-username` in the `git clone` URL with your GitHub username.
- Add the `requirements.txt` file to the repo that lists the dependencies (like `numpy`, `pandas`, etc.) if you don't already have it.

