# Handwriting Recognition iOS App

This directory contains the iOS version of the Handwriting Recognition project, which uses a CNN model trained on the EMNIST dataset to recognize handwritten characters.
<br>
<div align="center">
  <img src="https://cdn-icons-png.flaticon.com/512/5968/5968371.png" alt="Swift Logo" height="64">
  <span style="display:inline-block; width: 3%;"></span>
  <img src="https://logos-world.net/wp-content/uploads/2023/06/iOS-Symbol.png" alt="iOS Logo" height="64">
</div>

## Project Structure

```
HandwritingRecognition/
├── ContentView.swift          # Main UI implementation
├── DrawingPredictor.swift     # Core ML model integration
├── HandwritingRecognitionApp.swift  # App entry point
├── convert_to_coreml.py       # Script to convert TensorFlow model to Core ML
└── saved_coreml_models/             # Directory containing the Core ML model
    └── HandwritingRecognition_CNN.mlmodel
```

## Features

- Real-time handwriting recognition
- Clean and intuitive drawing interface using PencilKit
- Support for Apple Pencil (on compatible devices)
- Instant character prediction using Core ML

## Prerequisites

To build and run this iOS app, you'll need:

- A Mac computer running macOS
- Xcode 14.0 or later
- iOS 15.0 or later on the target device
- (Optional) Apple Developer account for deploying to physical devices

## Requirements

- iOS 15.0+
- Xcode 14.0+
- Swift 5.0+
- Core ML 3+

## Getting Started
1. Clone the repository

2. Convert the models to Core ML format:
- The original CNN model (`cnn.h5`) is trained using TensorFlow. 
- Using CoreMLTools, the model is converted to Core ML format
   ```bash
   pip install coremltools tensorflow
   python convert_to_coreml.py
   ```
- The converted model is saved as `HandwritingRecognition_CNN.mlmodel`

3. Create a new Xcode project:
   - Open Xcode
   - Create a new iOS App project
   - Choose SwiftUI for the interface
   - Name it "HandwritingRecognition"

4. Add the files to your project:
   - Copy `ContentView.swift` to your project
   - Copy `DrawingPredictor.swift` to your project
   - Drag the generated `HandwritingRecognition.mlmodel` into your Xcode project

5. Build and run the project in Xcode

## Usage

1. Launch the app
2. Draw a character in the drawing area
3. Tap "Predict" to get the recognition result
4. Use "Clear" to reset the drawing area

## Technical Details

- Input: 28x28 grayscale image
- Output: Predicted character
- Model: Convolutional Neural Network (CNN)
- Framework: Core ML, Vision, SwiftUI, PencilKit

## Integration Notes

This iOS app is part of a larger project that includes:
- Python-based model training
- Gradio web interface
- Model conversion utilities

The iOS version provides native mobile support with the same underlying CNN model used in the web version.


