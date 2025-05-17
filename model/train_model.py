import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

class PaddySeedClassifier:
    def __init__(self, img_size=(224, 224), batch_size=32, learning_rate=1e-4):
        self.img_size = img_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = None

    def create_model(self):
        base_model = ResNet50(weights='imagenet', include_top=False,
                              input_shape=(self.img_size[0], self.img_size[1], 3))
        base_model.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x)
        output = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=base_model.input, outputs=output)

        model.compile(optimizer=Adam(learning_rate=self.learning_rate),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        self.model = model
        return model

    def prepare_data(self, dataset_path):
        print(f"Checking dataset path: {dataset_path}")
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

        print("Dataset folders:")
        for folder in os.listdir(dataset_path):
            print(f" - {folder}")

        # Custom preprocessing function for brightness increase
        def preprocess_brightness(image):
            image = tf.image.random_brightness(image, max_delta=0.2)
            return image

        datagen = ImageDataGenerator(
            preprocessing_function=preprocess_brightness,
            rescale=1./255,
            validation_split=0.15,
            horizontal_flip=True,
            rotation_range=20,
            zoom_range=0.15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            fill_mode='nearest'
        )

        train_gen = datagen.flow_from_directory(
            dataset_path,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            subset='training',
            shuffle=True
        )

        val_gen = datagen.flow_from_directory(
            dataset_path,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            subset='validation',
            shuffle=True
        )

        return train_gen, val_gen

    def train(self, dataset_path, epochs=10):
        train_gen, val_gen = self.prepare_data(dataset_path)

        history = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            verbose=1
        )

        self.save_model()
        return history

    def save_model(self, save_path='seed_classifier_resnet50.h5'):
        self.model.save(save_path)
        print(f'Model saved to {save_path}')


if __name__ == '__main__':
    # Update this path to your dataset directory containing class subfolders
    dataset_path = r'C:\Users\Aksha\Desktop\seed_quality\dataset\train'

    classifier = PaddySeedClassifier()
    classifier.create_model()
    classifier.train(dataset_path, epochs=10)
