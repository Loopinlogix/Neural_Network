# Neural_Network
%%writefile README.md

# Handwritten Digit Classification using a Simple Neural Network (MNIST Dataset)

## Introduction

This project dives into applying neural networks to a classic image classification problem: recognizing handwritten digits. The main goal was to build, train, and evaluate a straightforward feedforward neural network on the famous MNIST dataset. It's all about getting some hands-on experience with how neural networks work for image classification and seeing why deep learning is often favored over older machine learning methods for such tasks.

## Project Overview & Methodology

I built this project following several key steps to classify handwritten digits (0-9) from the MNIST dataset:

1.  **Data Acquisition and Preparation**: Loaded the MNIST dataset (70,000 grayscale images, 28x28 pixels) using `fetch_openml`. Features (`X`) and labels (`y`) were set to `float32` and `int` types, respectively.

2.  **Data Splitting**: Divided the dataset into an 80% training set and a 20% test set using `train_test_split`, ensuring an even distribution of digit classes across both sets.

3.  **Preprocessing**: Essential steps included:
    *   **Reshaping**: Transformed the 1D pixel data (784 features) into a 2D image format (`28x28x1`) as expected by Keras.
    *   **Normalization**: Scaled pixel values from [0, 255] to [0, 1] for more stable neural network training.
    *   **One-Hot Encoding**: Converted integer labels (0-9) into a one-hot encoded format (e.g., '5' becomes `[0,0,0,0,0,1,0,0,0,0]`), necessary for the chosen loss function.

4.  **Neural Network Architecture**: Designed a `tf.keras.models.Sequential` model with:
    *   An `Input` layer to define the image shape.
    *   A `Flatten` layer to convert 2D images to 1D vectors.
    *   Two `Dense` hidden layers (128 and 64 neurons) with `relu` activation.
    *   A `Dense` output layer (10 neurons for 10 classes) with `softmax` activation for probability distribution.

5.  **Model Compilation**: Configured the model with the `adam` optimizer, `categorical_crossentropy` as the loss function (for multi-class classification with one-hot labels), and `accuracy` as the evaluation metric.

6.  **Model Training**: Trained the model using `model.fit()` with a `batch_size` of 32 and for 15 `epochs`. A `validation_split` of 0.1 was used to monitor performance on unseen data during training and check for overfitting.

7.  **Model Evaluation**: Assessed the model's performance on the test set using:
    *   **Test Accuracy**: A direct measure of correct predictions.
    *   **Classification Report**: Detailed metrics (precision, recall, F1-score) for each digit class.
    *   **Confusion Matrix**: A visual representation (heatmap) of correct vs. incorrect classifications, highlighting specific misclassifications.
    *   **Training Curves**: Plots showing accuracy and loss over epochs for both training and validation sets.

## Results Analysis

The neural network achieved an impressive **97.75% accuracy** on the test set. Training and validation curves showed good convergence with minimal overfitting. The classification report indicated strong performance across all digit classes, and the confusion matrix revealed only minor, rare misclassifications (e.g., some '2's mistaken for '7's or '8's).

## Setup and How to Run

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Loopinlogix/Neural_Network.git
    cd Neural_Network
    ```
2.  **Open in Google Colab:** Upload the `.ipynb` file to Google Colab.
3.  **Run All Cells:** Execute all cells in the notebook to reproduce the results.

## Libraries Used

*   `numpy`
*   `pandas`
*   `scikit-learn`
*   `tensorflow` (with `Keras`)
*   `matplotlib`
*   `seaborn`
