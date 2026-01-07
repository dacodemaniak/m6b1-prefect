import tensorflow as tf
from tensorflow.keras import layers, models
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / 'models' / 'model_mnist.keras'

def create_and_save_base_model():
    # 1. Chargement du dataset MNIST standard
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # 2. Prétraitement (Normalisation entre 0 et 1)
    x_train = x_train.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
    x_test = x_test.reshape((-1, 28, 28, 1)).astype('float32') / 255.0

    # 3. Architecture du CNN
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ])

    # 4. Compilation
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # 5. Entraînement rapide (5 epochs suffisent pour la V0)
    print("Entraînement du modèle initial...")
    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

    # 6. Sauvegarde au format Keras
    model.save(str(MODEL_PATH))
    print("Fichier 'model_mnist.keras' généré avec succès !")

if __name__ == "__main__":
    create_and_save_base_model()