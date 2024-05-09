import tensorflow as tf
import numpy as np

# Assuming each input feature vector has 784 features
model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dropout(rate=0.5)
])

# Example synthetic data for illustration
x_train = np.random.random((60000, 784))
y_train = np.random.random((60000,))
x_val = np.random.random((10000, 784))
y_val = np.random.random((10000,))

model.compile(optimizer='adam', loss='mse')
model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))

model.save('my_model.h5')
