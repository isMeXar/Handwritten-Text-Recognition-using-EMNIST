import coremltools as ct
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

def convert_cnn():
    print("Loading CNN model...")
    # Load the CNN model
    keras_model = load_model('../../saved/cnn.h5')
    
    # Define the input shape for Core ML (remove the batch size 'None')
    coreml_input_shape = (1, 28, 28, 1)  # Assuming a single grayscale image of size 28x28
    
    print("Converting CNN model to Core ML...")
    # Convert the Keras model to Core ML
    coreml_model = ct.convert(
        keras_model,
        inputs=[ct.ImageType(shape=coreml_input_shape)]
    )
    
    print("Saving CNN model...")
    # Save the Core ML model
    coreml_model.save("HandwritingRecognition_CNN.mlmodel")
    print("CNN model conversion completed!")

def main():
    print("Starting model conversion process...")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"CoreML Tools version: {ct.__version__}")
    
    try:
        convert_cnn()
    except Exception as e:
        print(f"Error converting CNN model: {str(e)}")
    
    print("Model conversion process completed!")

if __name__ == "__main__":
    main()
