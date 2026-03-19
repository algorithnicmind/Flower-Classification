import os
import pathlib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import shutil

# Configuration
DATASET_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
BATCH_SIZE = 32
IMG_HEIGHT = 180
IMG_WIDTH = 180
EPOCHS = 15
FINE_TUNE_EPOCHS = 3  # Epochs for learning from a single new image

# Paths
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DATA_DIR = os.path.join(PROJECT_DIR, "training_data")
USER_UPLOADS_DIR = os.path.join(PROJECT_DIR, "user_uploads")
MODEL_PATH = os.path.join(PROJECT_DIR, "model.tflite")
KERAS_MODEL_PATH = os.path.join(PROJECT_DIR, "model.keras")
CLASSES_PATH = os.path.join(PROJECT_DIR, "classes.txt")


def setup_training_data():
    """Download the original dataset and set up the training_data folder."""
    
    # If training_data already exists with images, skip download
    if os.path.exists(TRAINING_DATA_DIR):
        image_count = len(list(pathlib.Path(TRAINING_DATA_DIR).glob('*/*.jpg')))
        if image_count > 0:
            print(f"Training data already exists with {image_count} images. Skipping download.")
            return TRAINING_DATA_DIR
    
    # Download the dataset
    print("Downloading dataset for the first time...")
    data_dir_tar = tf.keras.utils.get_file('flower_photos.tar', origin=DATASET_URL, extract=True)
    data_dir = pathlib.Path(data_dir_tar).with_suffix('')
    
    # Handle nested folder structure
    if (data_dir / 'flower_photos').exists():
        data_dir = data_dir / 'flower_photos'
    
    # Copy to our local training_data folder
    print(f"Copying dataset to {TRAINING_DATA_DIR}...")
    if os.path.exists(TRAINING_DATA_DIR):
        shutil.rmtree(TRAINING_DATA_DIR)
    shutil.copytree(str(data_dir), TRAINING_DATA_DIR)
    
    # Remove the LICENSE file if it exists (it's not an image)
    license_file = os.path.join(TRAINING_DATA_DIR, "LICENSE.txt")
    if os.path.exists(license_file):
        os.remove(license_file)
    
    image_count = len(list(pathlib.Path(TRAINING_DATA_DIR).glob('*/*.jpg')))
    print(f"Dataset ready with {image_count} images.")
    return TRAINING_DATA_DIR


def train(data_dir):
    """Full training of the model on the entire dataset (run only once)."""
    
    data_path = pathlib.Path(data_dir)
    image_count = len(list(data_path.glob('*/*.jpg')))
    print(f"\nTotal images for training: {image_count}")
    
    # Split into Training and Validation
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_path,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_path,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    class_names = train_ds.class_names
    print(f"Classes found: {class_names}")

    # Optimize for Performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Data Augmentation
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])

    # Build Model
    num_classes = len(class_names)
    model = Sequential([
        data_augmentation,
        layers.Rescaling(1./255),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, name="outputs")
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    model.summary()

    # Train the Model
    print(f"\nTraining for {EPOCHS} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )

    # Save the full Keras model (needed for fine-tuning later)
    model.save(KERAS_MODEL_PATH)
    print(f"Keras model saved to {KERAS_MODEL_PATH}")

    # Convert and Save the TFLite model
    print("\nConverting model to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(MODEL_PATH, 'wb') as f:
        f.write(tflite_model)
    print(f"TFLite model saved to {MODEL_PATH}")

    # Save class names
    with open(CLASSES_PATH, 'w') as f:
        f.write("\n".join(class_names))
    print(f"Classes saved to {CLASSES_PATH}")

    return history


def fine_tune_on_new_image(image_path, class_name):
    """
    Fine-tune the EXISTING model on ONLY the new image.
    This does NOT re-train on old images. It only teaches
    the brain the new image.
    """
    
    print(f"\n🧠 Fine-tuning the AI on your new image only...")
    print(f"   Image: {image_path}")
    print(f"   Class: {class_name}")
    
    # 1. Load the existing Keras model (the full brain)
    if not os.path.exists(KERAS_MODEL_PATH):
        print("❌ Keras model not found. Please run 'python train_model.py' first.")
        return
    
    model = keras.models.load_model(KERAS_MODEL_PATH)
    
    # 2. Load class names to find the class index
    with open(CLASSES_PATH, 'r') as f:
        class_names = f.read().splitlines()
    
    class_index = class_names.index(class_name)
    
    # 3. Prepare ONLY the new image as training data
    img = tf.keras.utils.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = tf.keras.utils.img_to_array(img)
    
    # Create a small batch with just this one image
    img_batch = tf.expand_dims(img_array, 0)        # Shape: (1, 180, 180, 3)
    label_batch = tf.constant([class_index])          # Shape: (1,)
    
    # Create a dataset from just this image
    new_ds = tf.data.Dataset.from_tensor_slices((img_batch, label_batch))
    new_ds = new_ds.batch(1)
    
    # 4. Fine-tune with a LOW learning rate (so it doesn't forget old knowledge)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    print(f"   Training on new image for {FINE_TUNE_EPOCHS} epochs...")
    model.fit(new_ds, epochs=FINE_TUNE_EPOCHS, verbose=1)
    
    # 5. Save the updated Keras model
    model.save(KERAS_MODEL_PATH)
    print(f"   ✅ Updated Keras model saved.")
    
    # 6. Convert and save updated TFLite model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    with open(MODEL_PATH, 'wb') as f:
        f.write(tflite_model)
    print(f"   ✅ Updated TFLite model saved.")
    
    print(f"\n🎉 AI has learned from your new image! Brain updated.")


def main():
    # 1. Set up training data (download if first time)
    setup_training_data()
    
    # 2. Train the model (full training)
    train(TRAINING_DATA_DIR)
    
    print("\n✅ Training complete! Model is ready.")


if __name__ == "__main__":
    main()
