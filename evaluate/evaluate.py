import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random
from tensorflow.keras.utils import load_img, img_to_array

# Paths
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../model/seed_classifier_resnet50.h5')
TEST_DATA_DIR = r"C:\Users\Aksha\Desktop\seed_quality\test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def load_and_preprocess_image(img_path):
    img = load_img(img_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    return img_array

def main():
    print("Loading model...")
    model = load_model(MODEL_PATH)

    print("Preparing test data generator...")
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        TEST_DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )

    print("Evaluating model on full test set...")
    loss, accuracy = model.evaluate(test_generator)
    print(f"Overall Test Loss: {loss:.4f}")
    print(f"Overall Test Accuracy: {accuracy*100:.2f}%")

    # Get class indices mapping
    class_indices = test_generator.class_indices
    idx_to_class = {v: k for k, v in class_indices.items()}

    # Select 15 random images from each folder
    pure_dir = os.path.join(TEST_DATA_DIR, 'pure')
    impure_dir = os.path.join(TEST_DATA_DIR, 'impure')

    pure_imgs = [os.path.join(pure_dir, f) for f in os.listdir(pure_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    impure_imgs = [os.path.join(impure_dir, f) for f in os.listdir(impure_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    sample_pure = random.sample(pure_imgs, min(15, len(pure_imgs)))
    sample_impure = random.sample(impure_imgs, min(15, len(impure_imgs)))

    sample_imgs = sample_pure + sample_impure
    true_labels = [class_indices['pure']] * len(sample_pure) + [class_indices['impure']] * len(sample_impure)

    # Load and preprocess images for prediction
    X_sample = np.array([load_and_preprocess_image(img) for img in sample_imgs])

    # Predict
    preds = model.predict(X_sample)
    pred_classes = (preds > 0.5).astype('int32').flatten()

    # Calculate accuracy on the sample set
    sample_accuracy = np.mean(pred_classes == true_labels)
    print(f"Sampled subset accuracy (30 images): {sample_accuracy*100:.2f}%")

    # Visualize predictions
    plt.figure(figsize=(18, 8))
    for i, img_path in enumerate(sample_imgs):
        img = plt.imread(img_path)
        true_label = true_labels[i]
        pred_label = pred_classes[i]
        pred_prob = preds[i][0]

        plt.subplot(3, 10, i+1)
        plt.imshow(img)
        plt.axis('off')
        title_color = 'green' if true_label == pred_label else 'red'
        plt.title(
            f"True: {idx_to_class[true_label]}\nPred: {idx_to_class[pred_label]}\nConf: {pred_prob:.2f}",
            color=title_color,
            fontsize=9
        )
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
