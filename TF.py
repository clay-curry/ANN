from turtle import color
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
np.set_printoptions(precision=3,suppress=True)

samples=50
signal = lambda x : x**4 - 22*x**2
# signal = lambda x : 4*x + 3
noise=np.random.uniform(-10,10,(samples,))

features = np.random.uniform(low=(-5.0), high=(5.0), size=(samples,))
labels = signal(features)
labels = labels + noise

model = tf.keras.Sequential([
    layers.Input(shape=(1,)),
    layers.Dense(units=64, activation='relu'),
    layers.Dense(units=64, activation='relu'),
    layers.Dense(units=1)
])
model.summary()
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.1),  loss='mean_absolute_error')
history = model.fit(
    features,
    labels,
    epochs=1000,
    verbose=0,
    validation_split = 0.2
)


plot = np.linspace(-5,5,90)
pred = model.predict(plot)
signal = signal(plot)

plt.scatter(features, labels)
plt.plot(plot,pred,c='#88c999')
plt.plot(plot,signal,'r')
plt.xlabel('Argument')
plt.ylabel('Expected output')
plt.title('Function Approximations using Neural Networks')
plt.text(-3,50, r'$')
plt.show()


