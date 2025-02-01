import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
    BatchNormalization, InputLayer, AveragePooling2D, GlobalAveragePooling2D, Rescaling
)
from keras.optimizers import Adam
from keras.metrics import Precision, Recall, AUC, Accuracy, BinaryAccuracy
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.applications import EfficientNetB0

import matplotlib.pyplot as plt

class ParticleClassifier:
    def __init__(self, model_type="lenet5", input_shape=(32, 32, 2)):
        self.input_shape = input_shape
        self.model_type = model_type
        self.model = None
        self.history = None
        self.build_model()

    def build_model(self):
        self.model = self.create_Lenet5_particle_classifier()
        # if self.model_type.lower() == "lenet5":
        #     self.model = self.create_Lenet5_particle_classifier()
        # elif self.model_type.lower() == "custom":
        #     self.model = self.create_custom_particle_classifier()
        # else:
        #     self.model = self.create_efficientnet_classifier()


    def create_Lenet5_particle_classifier(self, input_shape=(32, 32, 2)):
        model = Sequential()
        model.add(InputLayer(input_shape=input_shape))
        # Convolutional layers
        model.add(Conv2D(filters=6, kernel_size=(5, 5), strides=(1,1), activation='relu'))
        model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
        model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
        # Fully connected layers
        model.add(Flatten())
        model.add(Dense(units=120, activation='relu'))
        model.add(Dense(units=84,activation='relu', use_bias=True))
        model.add(Dense(units=1, activation='sigmoid'))
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=[
                Precision(name='precision'),
                Recall(name='recall'),
                AUC(name='auc'),
                BinaryAccuracy(threshold=0.5, name='accuracy')
            ]
        )
        return model

    def create_custom_particle_classifier(self, input_shape=(32, 32, 2)):
        model = Sequential([
            InputLayer(input_shape=input_shape),
            # Convolutional layers
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            # Fully connected layers
            Flatten(),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])

        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=[
                Precision(name='precision'),
                Recall(name='recall'),
                AUC(name='auc'),
                Accuracy(name='accuracy')
            ]
        )

        return model

    def create_efficientnet_classifier(self, input_shape=(32, 32, 2)):
        base_model = EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=(32, 32, 3)
        )

        # Create our model
        model = Sequential([
            # Input layer
            InputLayer(input_shape=self.input_shape),

            # Convert 2 channels to 3 channels
            Conv2D(3, (1, 1), padding='same'),

            # Rescale inputs to [-1, 1] as expected by EfficientNet
            Rescaling(scale=2.0, offset=-1.0),

            # Add the base model
            base_model,

            # Add top layers
            GlobalAveragePooling2D(),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])

        # Freeze the base model
        base_model.trainable = False

        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=[
                Precision(name='precision'),
                Recall(name='recall'),
                AUC(name='auc'),
                Accuracy(name='accuracy')
            ]
        )

        return model

    def get_callbacks(self, patience=10):
        """
        Define callbacks for training
        """
        callbacks = [
            # Stop training when validation loss stops improving
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),

            # Reduce learning rate when learning stagnates
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.1,
                patience=patience//2,
                verbose=1,
                min_lr=1e-6
            ),

            # Save the best model
            ModelCheckpoint(
                f'{self.model_type}.keras',
                monitor='val_auc',
                mode='max',
                save_best_only=True,
                verbose=1
            )
        ]
        return callbacks

    def train(self, train_dataset, validation_dataset, epochs=50, patience=10):
        self.history = self.model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=epochs,
            callbacks=self.get_callbacks(),
            verbose=1
        )

    def evaluate(self, test_dataset):
        results = self.model.evaluate(test_dataset, verbose=1)
        metrics = {name: value for name, value in zip(self.model.metrics_names, results)}
        return metrics

    def plot_history(self):
        if self.history is None:
            return
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))

        ax[0].plot(self.history.history['loss'], label='Training Loss')
        ax[0].plot(self.history.history['val_loss'], label='Validation Loss')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Loss')
        ax[0].legend()

        ax[1].plot(self.history.history['accuracy'], label='Training Accuracy')
        ax[1].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Accuracy')
        ax[1].legend()

        plt.tight_layout()
        plt.show()


    def save_model(self):
        self.model.save(f'{self.model_type}.keras')

    def load_model(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        return self.model

    def summary_model(self):
        self.model.summary()