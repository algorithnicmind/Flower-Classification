import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import PIL
import sys

# Configuration
TFLite_MODEL_FILE = 'model.tflite'
IMG_HEIGHT = 180
IMG_WIDTH = 180

def classify_local_image(image_path):
    print(f"Classifying: {image_path}")
    
    # 1. Load the interpreter and classes
    interpreter = tf.lite.Interpreter(model_path=TFLite_MODEL_FILE)
    
    # Check for signatures, usually 'serving_default'
    classify_lite = interpreter.get_signature_runner('serving_default')
    
    # Load class names
    if not os.path.exists('classes.txt'):
        print("classes.txt not found. Please train the model first.")
        return
    with open('classes.txt', 'r') as f:
        class_names = f.read().splitlines()
    
    # 2. Prepare the image
    img = tf.keras.utils.load_img(
        image_path, target_size=(IMG_HEIGHT, IMG_WIDTH)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    
    # 3. Model Inference
    # Get the input name from the signature list to ensure compatibility
    signature_list = interpreter.get_signature_list()
    input_name = list(signature_list['serving_default']['inputs'].keys())[0]
    
    predictions_lite_dict = classify_lite(**{input_name: img_array})
    predictions_lite = list(predictions_lite_dict.values())[0]
    
    # 4. Result with Softmax
    score_lite = tf.nn.softmax(predictions_lite)
    
    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score_lite)], 100 * np.max(score_lite))
    )

def download_sunflower():
    # Helper to download a test image
    sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
    print("Downloading test image (sunflower)...")
    sunflower_path = tf.keras.utils.get_file('Red_sunflower.jpg', origin=sunflower_url)
    return sunflower_path

if __name__ == "__main__":
    if len(sys.argv) > 1:
        classify_local_image(sys.argv[1])
    else:
        # Default with the sunflower image
        sunflower_path = download_sunflower()
        classify_local_image(sunflower_path)
