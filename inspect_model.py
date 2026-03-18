import tensorflow as tf

def inspect_tflite_model(model_path):
    print(f"Inspecting TFLite model: {model_path}\n")
    
    # Load the model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # 1. Input Details
    print("--- Input Tensors ---")
    for detail in interpreter.get_input_details():
        print(f"Name: {detail['name']}")
        print(f"Shape: {detail['shape']}")
        print(f"Type: {detail['dtype']}")
        print("-" * 20)
    
    # 2. Output Details
    print("\n--- Output Tensors ---")
    for detail in interpreter.get_output_details():
        print(f"Name: {detail['name']}")
        print(f"Shape: {detail['shape']}")
        print(f"Type: {detail['dtype']}")
        print("-" * 20)

    # 3. Model Signature
    print("\n--- Signature List ---")
    print(interpreter.get_signature_list())

    # 4. Total Tensors
    print(f"\nTotal Number of Tensors in model: {len(interpreter.get_tensor_details())}")

if __name__ == "__main__":
    inspect_tflite_model('model.tflite')
