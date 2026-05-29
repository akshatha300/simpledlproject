# Handwritten Digit Classifier using TensorFlow

A simple Deep Learning model built with TensorFlow/Keras to classify handwritten digits (0-9) using the classic MNIST dataset. 

## Project Architecture
- **Foundational Math**: Linear transformations scaled down to 0-1 pixel ranges.
- **Model Type**: Sequential Feedforward Neural Network.
- **Layers**: Flatten Input Layer -> Dense Hidden Layer (128 units, ReLU) -> Dense Output Layer (10 units, Softmax).

## Performance
- **Training Epochs**: 3
- **Test Accuracy**: ~97%

## How to Run It Locally
1. Clone the repository: `git clone https://github.com`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the script: `python train_classifier.py`
