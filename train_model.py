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

# Paths
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DATA_DIR = os.path.join(PROJECT_DIR, "training_data")
USER_UPLOADS_DIR = os.path.join(PROJECT_DIR, "user_uploads")
MODEL_PATH = os.path.join(PROJECT_DIR, "model.tflite")
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


def merge_user_uploads():
    """Copy any user-uploaded images into the training_data folder."""
    
    if not os.path.exists(USER_UPLOADS_DIR):
        print("No user uploads found. Skipping merge.")
        return 0
    
    new_images_count = 0
    for class_folder in os.listdir(USER_UPLOADS_DIR):
        src_folder = os.path.join(USER_UPLOADS_DIR, class_folder)
        dst_folder = os.path.join(TRAINING_DATA_DIR, class_folder)
        
        if not os.path.isdir(src_folder):
            continue
        
        # Create destination folder if it doesn't exist
        os.makedirs(dst_folder, exist_ok=True)
        
        for img_file in os.listdir(src_folder):
            src_path = os.path.join(src_folder, img_file)
            dst_path = os.path.join(dst_folder, img_file)
            
            if not os.path.exists(dst_path):
                shutil.copy2(src_path, dst_path)
                new_images_count += 1
    
    if new_images_count > 0:
        print(f"Merged {new_images_count} new user-uploaded images into training data.")
    else:
        print("No new user uploads to merge.")
    
    return new_images_count


def train(data_dir):
    """Train the model on the given data directory."""
    
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

    # Convert and Save the Model
    print("\nConverting model to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(MODEL_PATH, 'wb') as f:
        f.write(tflite_model)
    print(f"Model saved to {MODEL_PATH}")

    # Save class names
    with open(CLASSES_PATH, 'w') as f:
        f.write("\n".join(class_names))
    print(f"Classes saved to {CLASSES_PATH}")

    return history


def main():
    # 1. Set up training data (download if first time)
    setup_training_data()
    
    # 2. Merge any user uploads into training data
    merge_user_uploads()
    
    # 3. Train the model
    train(TRAINING_DATA_DIR)
    
    print("\n✅ Training complete! Model is ready.")


if __name__ == "__main__":
    main()
