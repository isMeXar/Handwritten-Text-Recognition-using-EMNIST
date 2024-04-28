# Handwritten Text Recognition using EMNIST
<img src="[src/resources/img/logo_banner.png](https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.researchgate.net%2Ffigure%2FSamples-of-all-letters-and-digits-in-the-EMNIST-dataset_fig2_334957576&psig=AOvVaw3wvYX1mwG9nsE8d6cZN2vH&ust=1714430956842000&source=images&cd=vfe&opi=89978449&ved=0CBIQjRxqFwoTCNDBxJr_5YUDFQAAAAAdAAAAABAg)" alt="Header Image" style="width:100%;">
This project focuses on the recognition of handwritten characters using machine learning and deep learning algorithms. It utilizes the EMNIST dataset, which contains handwritten digits [0-9] and letters [A-Za-z]. The goal is to accurately classify and categorize these characters.
## Dataset Used
The EMNIST dataset, a widely used benchmark dataset for character recognition, is used in this project. It consists of grayscale images of handwritten characters, along with corresponding labels indicating the character class.
https://www.kaggle.com/datasets/crawford/emnist
## Requirements
1. Python
2. Jupyter Notebook
3. Sklearn
4. numpy
5. Pandas
6. Matplotlib
7. TensorFlow
8. Keras
9. OpenCV 
11. Gradio
    
## Preprocessing Steps
The dataset undergoes several preprocessing steps to prepare it for model training and testing. These steps include:
    • Verification of missing values, duplication removal, and grayscale value range validation
    • Separation of input features (pixel values) and target labels
    • Address the misalignment and orientation issues in the original EMNIST dataset, resulting in a reshaped dataset of size (131598, 28, 28).
    • Flattening multidimensional arrays, by reshaping them into two-dimensional input for traditional machine learning algorithms. This enables the algorithms to process the data in a compatible input format, which is (1,784)
Then we added extra preprocessing steps for our deep learning model:
    • One-hot encoding of categorical labels
    • Reshaping data into a 4-dimensional array (1, 28, 28, 1) to be compatible with our model input requirement.
# Machine Learning Algorithms
Several machine learning algorithms are employed for character recognition. 
The following algorithms are implemented with their respective source code and accuracy scores:
## Random Forest
```python
# Create the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100)

# Train the model on your training set
rf_model.fit(X_train, y_train)

# Predict the labels of your validation set
y_pred_rf = rf_model.predict(X_test)

# Calculate accuracy
accuracy_rf = accuracy_score(y_test, y_pred_rf)

# Calculate precision, recall, and f1 score
precision_rf, recall_rf, f1_rf, _ = precision_recall_fscore_support(y_test, y_pred_rf, average='weighted')

# Print the evaluation metrics
print("Accuracy:", accuracy_rf)
print("Precision:", precision_rf)
print("Recall:", recall_rf)
print("F1 Score:", f1_rf)

Accuracy: 0.8118541033434651
Precision: 0.8119493522475134
Recall: 0.8118541033434651
F1 Score: 0.8098937096654448
```
## K-Nearest Neighbors (KNN)

```python
# Create a KNN classifier with k=5
knn_model = KNeighborsClassifier(n_neighbors=5)

# Train the model on your training set
knn_model.fit(X_train, y_train)

# Predict the labels of your validation set
y_pred_knn = knn_model.predict(X_test)

# Calculate accuracy
accuracy_knn = accuracy_score(y_test, y_pred_knn)

# Calculate precision, recall, and f1 score
precision_knn, recall_knn, f1_knn, _ = precision_recall_fscore_support(y_test, y_pred_knn, average='weighted')

# Print the evaluation metrics
print("Accuracy:", accuracy_knn)
print("Precision:", precision_knn)
print("Recall:", recall_knn)
print("F1 Score:", f1_knn)

Accuracy: 0.78290273556231
Precision: 0.7931620867778456
Recall: 0.78290273556231
F1 Score: 0.7822322350312396
```

## Naive Bayes

```python
# Create a KNN classifier with k=5
nb_model = GaussianNB()

# Train the model on your training set
nb_model.fit(X_train, y_train)

# Predict the labels of your validation set
y_pred_nb = nb_model.predict(X_test)

# Calculate accuracy
accuracy_nb = accuracy_score(y_test, y_pred_nb)

# Calculate precision, recall, and f1 score
precision_nb, recall_nb, f1_nb, _ = precision_recall_fscore_support(y_test, y_pred_nb, average='weighted')

# Print the evaluation metrics
print("Accuracy:", accuracy_nb)
print("Precision:", precision_nb)
print("Recall:", recall_nb)
print("F1 Score:", f1_nb)

Accuracy: 0.2762917933130699
Precision: 0.42236357411110537
Recall: 0.2762917933130699
F1 Score: 0.24116615965406896
```

## Support Vector Machine (SVM)
```python
# Create an SVM model
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)

# Train the model on your training set
svm_model.fit(X_train, y_train)

# Predict the labels of your validation set
y_pred_svm = svm_model.predict(X_test)

# Calculate accuracy
accuracy_svm = accuracy_score(y_test, y_pred_svm)

# Calculate precision, recall, and f1 score
precision_svm, recall_svm, f1_svm, _ = precision_recall_fscore_support(y_test, y_pred_svm, average='weighted')

# Print the evaluation metrics
print("Accuracy:", accuracy_svm)
print("Precision:", precision_svm)
print("Recall:", recall_svm)
print("F1 Score:", f1_svm)

Accuracy: 0.8436930091185411
Precision: 0.8449859853316789
Recall: 0.8436930091185411
F1 Score: 0.8419189462167611
```
## Models Saving
Saving the trained models allows for its reuse and deployment in various applications without the need for retraining. 
This ensures consistent and efficient utilization of the trained models.
```python
# To save a model
['./saved/svm.pkl']

# To load the saved models
joblib.dump(rf_model, './saved/random_forest.pkl')
joblib.dump(knn_model, './saved/knn.pkl')
joblib.dump(nb_model, './saved/naive_bayes.pkl')
joblib.dump(svm_model, './saved/svm.pkl')

```
# Convolutional Neural Network (CNN)
This section demonstrates the implementation of a Convolutional Neural Network (CNN) for the recognition of handwritten characters. CNNs have shown exceptional performance in image recognition tasks, making them well-suited for character recognition from image data.

```python
cnn_model = Sequential()

cnn_model.add(layers.Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu', input_shape=(28, 28, 1)))
cnn_model.add(layers.MaxPool2D(strides=2))

cnn_model.add(layers.Conv2D(filters=48, kernel_size=(5,5), padding='valid', activation='relu'))
cnn_model.add(layers.MaxPool2D(strides=2))

cnn_model.add(layers.Flatten())

cnn_model.add(layers.Dense(256, activation='relu'))
cnn_model.add(layers.Dense(84, activation='relu'))

cnn_model.add(layers.Dropout(0.2))

cnn_model.add(layers.Dense(number_of_classes, activation='softmax'))

cnn_model.summary()
```
## Model Compilation
The CNN model is compiled with the categorical cross-entropy loss function, the Adam optimizer, and accuracy as the evaluation metric. The Adam optimizer efficiently adapts the learning rate for faster convergence.

## Callbacks
Three callbacks are employed to improve model performance and prevent overfitting during training:

EarlyStopping: This callback monitors the validation loss and stops training if it doesn't improve for 5 epochs, preventing overfitting.
ModelCheckpoint: This callback saves the best model during training based on validation loss for later use.
ReduceLROnPlateau: This callback reduces the learning rate if validation loss plateaus for 3 epochs, allowing the model to fine-tune.

## Model Fitting
The CNN model is trained on the training data for 20 epochs with a batch size of 32. The training progress is monitored using the validation data provided during model fitting. The callbacks are employed to control the training process and achieve the best possible model for the task.

```python
# Compile the CNN model with categorical cross-entropy loss, the specified optimizer, and accuracy metric
cnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Create an EarlyStopping callback to monitor validation loss and stop training if it doesn't improve for 5 epochs
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')

# Create a ModelCheckpoint callback to save the best model during training based on validation loss
mcp_save = ModelCheckpoint('./saved/cnn.h5', save_best_only=True, monitor='val_loss', verbose=1, mode='auto')

# Create a ReduceLROnPlateau callback to reduce the learning rate if validation loss plateaus for 3 epochs
RLP = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.2, min_lr=0.0001)

# Model Fitting
history = cnn_model.fit(X_train, y_train, epochs=20, batch_size, verbose=1, validation_data=(X_test, y_test), callbacks=[mcp_save, early_stopping, RLP])
```

The CNN model achieved an accuracy of approximately 88.92% on the validation set after training for 9 epochs. The early stopping callback prevented further training as the validation loss reached a limit, ensuring the model's optimal performance and avoiding overfitting.
