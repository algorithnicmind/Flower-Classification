import tensorflow as tf
interpreter = tf.lite.Interpreter(model_path='model.tflite')
print(interpreter.get_signature_list())
