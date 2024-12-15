# Model Files

Due to size constraints, model files are not included in the Git repository. Here's how to obtain them:

## Training the Models
1. Run `cnn_model.ipynb` to train and save the CNN model (`cnn.h5`)
2. Run `machine_learning_models.ipynb` to train and save traditional ML models:
   - `knn.pkl`
   - `naive_bayes.pkl`
   - `random_forest.pkl`
   - `svm.pkl`

## Converting to Core ML
1. After training the CNN model, run `convert_to_coreml.py` in the iOS directory:
```bash
cd iOS/HandwritingRecognition
python convert_to_coreml.py
```
This will create `HandwritingRecognition_CNN.mlmodel` in the `saved_coreml_models` directory.

## Pre-trained Models
If you want to use pre-trained models, you can download them from [Google Drive/Release Assets - Add link here].

Place the downloaded files in their respective directories:
- TensorFlow/ML models in `saved_models/`
- Core ML model in `iOS/HandwritingRecognition/saved_coreml_models/`
