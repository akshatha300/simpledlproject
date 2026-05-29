import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Load the built-in handwriting dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. Normalize pixel values (foundational math concept!) from 0-255 down to 0-1
x_train, x_test = x_train / 255.0, x_test / 255.0

# 3. Design the Deep Learning Architecture (Fundamental blocks)
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),          # Flattens 2D images to 1D vector (784 neurons)
    layers.Dense(128, activation='relu'),          # Hidden layer with ReLU activation function
    layers.Dense(10, activation='softmax')         # Output layer providing 10 probabilities (digits 0-9)
])

# 4. Compile the model with basic optimization fundamentals
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5. Train the model (Backpropagation and Gradient Descent at work!)
print("--- STARTING THE TRAINING PROCESS ---")
model.fit(x_train, y_train, epochs=3)

# 6. Evaluate accuracy using completely unseen test data
print("\n--- EVALUATING MODEL ON UNSEEN DATA ---")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nYour model successfully read handwritten numbers with {test_acc*100:.2f}% accuracy!")
