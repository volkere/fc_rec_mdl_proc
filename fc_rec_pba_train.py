import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

print("TensorFlow-Version:", tf.__version__)

model_path = "/Users/blaubaer/Projects/age_timeline/own_model/age_model.h5"

@tf.keras.utils.register_keras_serializable()
def mse_loss(y_true, y_pred):
    return tf.keras.losses.MeanSquaredError()(y_true, y_pred)  # FIX

def build_age_model():
    model = models.Sequential([
        layers.Input(shape=(64, 64, 3)),  
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1)  
    ])
    model.compile(optimizer='adam', loss=mse_loss, metrics=['mae'])
    return model

X_train = np.random.rand(100, 64, 64, 3)  
y_train = np.random.randint(20, 80, size=(100,))

model = build_age_model()
model.fit(X_train, y_train, epochs=5, batch_size=10)  

model.save(model_path)
print(f"✅ Modell wurde neu trainiert und gespeichert unter: {model_path}")

