import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import h5py
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ParticleDataPreprocessor:
    def __init__(self, data_dir='./'):
        """
        Initialize the preprocessor with data directory path
        """
        self.data_dir = Path(data_dir)
        self.logger = logging.getLogger(__name__)

    def load_particle_data(self, particle_type, file_path):
        """
        Load particle data from HDF5 file
        """
        try:
            with h5py.File(file_path, 'r') as f:
                X = f['X'][:]
                y = f['y'][:]
                self.logger.info(f"Loaded {particle_type} data: {X.shape}")
                return X, y
        except Exception as e:
            self.logger.error(f"Error loading {particle_type} data: {e}")
            raise

    def preprocess_data(self, electrons_path, photons_path, sample_fraction=1.0,
                       test_size=0.1, random_state=42, batch_size=32):
        """
        Complete preprocessing pipeline
        """
        # Load data
        self.logger.info("Starting data preprocessing...")
        electrons_X, electrons_y = self.load_particle_data('electrons', electrons_path)
        photons_X, photons_y = self.load_particle_data('photons', photons_path)

        # Concat data
        X = np.concatenate((electrons_X, photons_X), axis=0)
        y = np.concatenate((electrons_y, photons_y), axis=0)
        y = np.expand_dims(y, axis=1)

        # If you cannot afford to load full dataset
        if sample_fraction < 1.0:
            sample_size = int(len(X) * sample_fraction)
            indices = np.random.RandomState(random_state).choice(
                len(X), sample_size, replace=False
            )
            X = X[indices]
            y = y[indices]
            self.logger.info(f"Sampled {sample_size} examples")

        # Shuffle data
        indices = np.random.RandomState(random_state).permutation(len(X))
        X = X[indices]
        y = y[indices]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Convert to TensorFlow tensors
        X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
        X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
        y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
        y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

        # Create TensorFlow datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))\
            .shuffle(buffer_size=1000)\
            .batch(batch_size)\
            .prefetch(tf.data.AUTOTUNE)

        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))\
            .batch(batch_size)\
            .prefetch(tf.data.AUTOTUNE)

        self.logger.info(f"Training set shape: {X_train.shape}")
        self.logger.info(f"Test set shape: {X_test.shape}")

        return train_dataset, test_dataset, X_train.shape[1:]

    def validate_data(self, dataset):
        """
        Validate the preprocessed data
        """
        for batch_x, batch_y in dataset.take(1):
            self.logger.info(f"Batch X shape: {batch_x.shape}")
            self.logger.info(f"Batch y shape: {batch_y.shape}")
            self.logger.info(f"X dtype: {batch_x.dtype}")
            self.logger.info(f"y dtype: {batch_y.dtype}")
            self.logger.info(f"X range: [{tf.reduce_min(batch_x)}, {tf.reduce_max(batch_x)}]")
            self.logger.info(f"Unique y values: {np.unique(batch_y)}")