import os
import sys
import shutil
import numpy as np
import tensorflow as tf
import PIL
from datetime import datetime

# Configuration
IMG_HEIGHT = 180
IMG_WIDTH = 180

# Paths
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_DIR, "model.tflite")
CLASSES_PATH = os.path.join(PROJECT_DIR, "classes.txt")
USER_UPLOADS_DIR = os.path.join(PROJECT_DIR, "user_uploads")


def classify_image(image_path):
    """Classify an image using the TFLite model."""
    
    print(f"\n📸 Classifying: {image_path}")
    
    # 1. Check if model exists
    if not os.path.exists(MODEL_PATH):
        print("❌ model.tflite not found. Please run 'python train_model.py' first.")
        return None
    
    if not os.path.exists(CLASSES_PATH):
        print("❌ classes.txt not found. Please run 'python train_model.py' first.")
        return None
    
    # 2. Load the interpreter and class names
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    classify_lite = interpreter.get_signature_runner('serving_default')
    
    with open(CLASSES_PATH, 'r') as f:
        class_names = f.read().splitlines()
    
    # 3. Prepare the image
    img = tf.keras.utils.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    
    # 4. Run Inference
    signature_list = interpreter.get_signature_list()
    input_name = signature_list['serving_default']['inputs'][0]
    
    predictions_dict = classify_lite(**{input_name: img_array})
    predictions = list(predictions_dict.values())[0]
    
    # 5. Get results with Softmax
    score = tf.nn.softmax(predictions[0])
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)
    
    print(f"🌸 This image most likely belongs to **{predicted_class}** with a {confidence:.2f}% confidence.")
    
    return predicted_class, confidence


def save_to_uploads(image_path, predicted_class):
    """Save the user's image to the user_uploads folder under the predicted class."""
    
    # Create the folder: user_uploads/<predicted_class>/
    class_folder = os.path.join(USER_UPLOADS_DIR, predicted_class)
    os.makedirs(class_folder, exist_ok=True)
    
    # Create a unique filename using timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    original_name = os.path.basename(image_path)
    new_name = f"{timestamp}_{original_name}"
    dest_path = os.path.join(class_folder, new_name)
    
    # Copy the image (don't move it, user might still need it)
    shutil.copy2(image_path, dest_path)
    print(f"💾 Image saved to: {dest_path}")
    
    return dest_path


def download_sunflower():
    """Helper to download a test image."""
    sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
    print("Downloading test image (sunflower)...")
    sunflower_path = tf.keras.utils.get_file('Red_sunflower.jpg', origin=sunflower_url)
    return sunflower_path


def main():
    # Get the image path
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Default: download and use a test sunflower image
        image_path = download_sunflower()
    
    # Check if the image file exists
    if not os.path.exists(image_path):
        print(f"❌ Image file not found: {image_path}")
        return
    
    # Step 1: Classify the image
    result = classify_image(image_path)
    
    if result is None:
        return
    
    predicted_class, confidence = result
    
    # Step 2: Save the image to user_uploads folder
    saved_path = save_to_uploads(image_path, predicted_class)
    
    # Step 3: Fine-tune ONLY on this new image (not on old images)
    from train_model import fine_tune_on_new_image
    fine_tune_on_new_image(saved_path, predicted_class)


if __name__ == "__main__":
    main()
