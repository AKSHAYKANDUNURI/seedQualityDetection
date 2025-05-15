import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../model/seed_classifier_resnet50.h5')
model = load_model(MODEL_PATH)

# Dataset directory
DATASET_PATH = r'C:\Users\Aksha\Desktop\seed_quality\seed_quality\dataset\train'  # Update this path
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Prepare data generator for evaluation
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.15)
val_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# Evaluate model
loss, accuracy = model.evaluate(val_gen)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print(f"Model Loss: {loss:.4f}")

# Generate predictions
predictions = model.predict(val_gen)
y_true = val_gen.classes
y_pred = (predictions > 0.5).astype('int32').flatten()

# Classification report
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=['Bad Seed', 'Good Seed']))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Bad Seed', 'Good Seed'])

# Visualization
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Confusion matrix plot
display.plot(ax=ax[0], cmap='Blues')
ax[0].set_title("Confusion Matrix")

# Accuracy and Loss plot
epochs = range(len(predictions))
ax[1].plot(epochs, predictions, label='Predictions')
ax[1].set_title("Model Predictions")
ax[1].set_xlabel("Image Index")
ax[1].set_ylabel("Prediction Confidence")
ax[1].legend()

plt.tight_layout()
plt.show()
