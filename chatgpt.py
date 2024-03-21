import tensorflow as tf
from tensorflow.keras import layers

# Tworzenie modelu sekwencyjnego
model = tf.keras.Sequential()

# Dodawanie warstw
model.add(layers.Dense(64, activation='relu', input_shape=(100,)))  # Warstwa wejściowa
model.add(layers.Dense(64, activation='relu'))  # Ukryta warstwa
model.add(layers.Dense(10, activation='softmax'))  # Warstwa wyjściowa

# Kompilacja modelu
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Wyświetlenie struktury modelu
model.summary()
