# Handwritten Text Recognition using EMNIST
<img src="img/header_img.png" alt="Header Image" style="width:100%;">

This project focuses on the recognition of handwritten characters using machine learning and deep learning algorithms. It utilizes the EMNIST dataset, which contains handwritten digits [0-9] and letters [A-Za-z]. The project includes both a web interface using Gradio and an iOS application for mobile deployment.

## Project Components

1. **Web Application**
   - Interactive web interface using Gradio
   - Support for multiple ML models
   - Real-time character recognition

2. **iOS Application**
   - Native iOS app using SwiftUI and Core ML
   - Real-time drawing and recognition
   - Optimized CNN model for mobile

3. **Machine Learning Models**
   - CNN (Best performer: 89.04% accuracy)
   - Traditional ML models (SVM, Random Forest, KNN, Naive Bayes)

## Dataset Used
The <a href="https://www.kaggle.com/datasets/crawford/emnist" target="_blank">EMNIST Dataset</a>, a widely used benchmark dataset for character recognition, is used in this project. It consists of grayscale images of handwritten characters, along with corresponding labels indicating the character class.

## Features

- Multiple approaches to handwritten text recognition:
  - Deep Learning models (CNN)
  - Traditional Machine Learning models
- Cross-platform deployment:
  - Web interface using Gradio
  - iOS application using Core ML
- Interactive drawing interfaces
- Model conversion utilities (TensorFlow to Core ML)
- Comprehensive evaluation metrics

## Project Structure

```
├── gradio_application.ipynb     # Main Gradio web interface
├── cnn_model.ipynb             # CNN model implementation
├── machine_learning_models.ipynb # Traditional ML models
├── requirements.txt            # Python dependencies
├── img/                       # Documentation images
├── saved_models/              # Saved model files
│   ├── cnn.h5                # TensorFlow model
│   ├── knn.pkl              # KNN model
│   ├── naive_bayes.pkl      # Naive Bayes model
│   ├── random_forest.pkl    # Random Forest model
│   └── svm.pkl             # SVM model
└── iOS/                      # iOS application
    └── HandwritingRecognition/
        ├── ContentView.swift          # Main UI
        ├── DrawingPredictor.swift     # Model integration
        ├── HandwritingRecognitionApp.swift # App entry point
        ├── HandwritingRecognitionTests.swift # Unit tests
        ├── Info.plist                 # App configuration
        ├── convert_to_coreml.py       # Model conversion
        └── saved_coreml_models/       # iOS model files
            └── HandwritingRecognition_CNN.mlmodel        # Core ML model
```

## Requirements

### Python Environment
- Python 3.x
- TensorFlow 2.x
- OpenCV
- scikit-learn
- NumPy
- Pandas
- Gradio
- Jupyter Notebook

### iOS Development (Optional)
- macOS with Xcode 14.0+
- iOS 15.0+
- Swift 5.0+
- Core ML 3+

# Machine Learning Models Performance
Several machine learning algorithms are employed for character recognition. 
The following algorithms are implemented with their respective source code and accuracy scores:

## Traditional ML Models

| Model               | Accuracy | Precision | Recall | F1 Score |
|--------------------|----------|-----------|---------|-----------|
| SVM                | 84.37%   | 84.50%    | 84.37%  | 84.19%    |
| Random Forest      | 81.19%   | 81.19%    | 81.19%  | 80.99%    |
| KNN                | 78.29%   | 79.32%    | 78.29%  | 78.22%    |
| Naive Bayes        | 27.63%   | 42.24%    | 27.63%  | 24.12%    |


## Convolutional Neural Network (CNN)
This section demonstrates the implementation of a Convolutional Neural Network (CNN) for the recognition of handwritten characters. CNNs have shown exceptional performance in image recognition tasks, making them well-suited for character recognition from image data.


### Model Compilation
The CNN model is compiled with the categorical cross-entropy loss function, the Adam optimizer, and accuracy as the evaluation metric. The Adam optimizer efficiently adapts the learning rate for faster convergence.
```python
cnn_model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss="categorical_crossentropy",
              metrics=["accuracy"])
```

### Callbacks
Three callbacks are employed to improve model performance and prevent overfitting during training:

- EarlyStopping: This callback monitors the validation loss and stops training if it doesn't improve for 5 epochs, preventing overfitting.
- ModelCheckpoint: This callback saves the best model during training based on validation loss for later use.
- ReduceLROnPlateau: This callback reduces the learning rate if validation loss plateaus for 3 epochs, allowing the model to fine-tune.


### Building the Model

```python
cnn_model = Sequential([
    layers.Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPool2D(strides=2),
    
    layers.Conv2D(filters=48, kernel_size=(5, 5), padding='valid', activation='relu'),
    layers.MaxPool2D(strides=2),
    
    layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'),
    layers.MaxPool2D(strides=2),
    
    layers.Flatten(),
    
    layers.Dense(256, activation='relu'),
    layers.Dense(84, activation='relu'),
    
    layers.Dropout(0.2),
    
    layers.Dense(number_of_classes, activation='softmax')
])
```

### Model Evaluation
The CNN model performed really well, here is the result:

```python
Best accuracy: 0.8904
Best loss value: 0.2377
```
<img src="img/CNN evaluation.png" alt="Header Image" style="width:100%;">


The CNN model achieved an accuracy of approximately 89.04% on the validation set after training for 9 epochs. The early stopping callback prevented further training as the validation loss reached a limit, ensuring the model's optimal performance and avoiding overfitting.


## Usage

### Installation

1. Clone this repository:
```bash
git clone https://github.com/isMeXar/Handwritten-Text-Recognition-using-EMNIST.git
```
2. Create and activate a virtual environment:
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```
3. Install the required packages

### Web Interface
1. Open Jupyter Notebook:
```bash
jupyter notebook
```

2. Navigate to `src/application.ipynb` to run the main application
3. For model training and experimentation:
   - `cnn_model.ipynb`: CNN model implementation and training
   - `machine_learning_models.ipynb`: Traditional ML approaches
   - `cross_val.ipynb`: Model evaluation and cross-validation

### IOS Application
1. Go to the `iOS` directory
3. Follow instruction in [iOS/HandwritingRecognition/README.md](iOS/HandwritingRecognition/README.md)
3. Build and run the iOS application


## Conclusion

This project demonstrates the effectiveness of different machine learning approaches for handwritten text recognition:

- The CNN model achieved the best performance with 89.04% accuracy, showcasing the power of deep learning for image recognition tasks
- Traditional ML models like SVM (84.37%) and Random Forest (81.19%) also showed strong performance
- The comprehensive preprocessing pipeline and model evaluation framework ensure robust and reliable results
- The interactive application interface makes the models accessible for practical use

Future improvements could include:
- Data augmentation techniques
- More complex CNN architectures
- Continuous text recognition
- Android application support
- Real-time recognition improvements
