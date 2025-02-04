# Particle Classification using Deep Learning

This project implements the LeNet-5 architecture to classify particle physics data from CERN, specifically distinguishing between electrons and photons using their detector signatures. The implementation focuses on binary classification of particle detector measurements using deep learning techniques.

## Dataset Description

The dataset consists of particle detector measurements from CERN. These measurements are fundamental for particle identification in high-energy physics experiments, allowing us to distinguish between different types of particles based on their interaction patterns with the detector.

### Example Particle Signatures

#### Electrons
![Electron Energy channel example](https://github.com/user-attachments/assets/e59831cb-1394-4892-8b71-00e8b8c06a6d)![Electron Time channel example](https://github.com/user-attachments/assets/64c41e8c-28dc-458c-8123-e5e1fcce6bc2)\
Electron signatures typically show more diffuse energy deposits across the detector. The image above shows electron interactions in both energy (left) and timing (right) channels. Notice how the energy deposits tend to spread out more compared to photons.

#### Photons
![Photon Energy channel example](https://github.com/user-attachments/assets/74f9be5e-60f2-47a6-9549-95d57ce37d59)![Photon Time channel example](https://github.com/user-attachments/assets/64bd32f1-1afd-42c3-add0-6c9a696f61a9)

Photon signatures characteristically display more concentrated energy patterns. The visualization above demonstrates photon interactions, showing their typically more localized energy deposits in both channels.

### Data Characteristics

- **Source**: CERN's particle detector measurements
- **Format**: Two HDF5 files (one for electrons, one for photons)
- **Size**: 249,000 images per particle type (498,000 total)
- **Image Dimensions**: 32x32 pixels
- **Channels**: 2 channels per image
  - Channel 1: Energy deposits
  - Channel 2: Timing information
- **Class Balance**: Equal distribution (50% electrons, 50% photons)

### Data Structure

Each image represents a detector "snapshot" of a particle interaction. The dual-channel structure captures different aspects of particle behavior:
- Energy Channel: Records the spatial distribution of energy deposits
- Timing Channel: Captures temporal aspects of the particle interaction

## Model Architecture: LeNet-5

The project adapts the LeNet-5 architecture for binary classification of particle images:

```python
model = Sequential([
    InputLayer(input_shape=(32, 32, 2)),
    Conv2D(filters=6, kernel_size=(5, 5), activation='relu'),
    AveragePooling2D(pool_size=(2, 2)),
    Conv2D(filters=16, kernel_size=(5, 5), activation='relu'),
    AveragePooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(units=120, activation='relu'),
    Dense(units=84, activation='relu'),
    Dense(units=1, activation='sigmoid')
])
```

### Training Configuration

The model training uses the following parameters:
- Optimizer: Adam (learning rate = 0.001)
- Loss Function: Binary Cross-entropy
- Batch Size: 32
- Early Stopping: Patience = 10 epochs
- Learning Rate Reduction: Factor = 0.1, Patience = 5 epochs
- Data Split: 90% training, 10% testing
- Training Sample: 20% of total dataset (due to computational constraints)

## Results and Analysis

### Training History
![Training History](https://github.com/user-attachments/assets/9ebf395b-a6f7-434a-aaa4-67599ec4306c)

The training process showed:
- Convergence after approximately 30 epochs
- Final training accuracy: 71.21%
- No significant overfitting (early stopping proved effective)
- Stable learning curve with occasional plateaus

### Performance Metrics

1. Overall Performance:
   - Accuracy: 71.21%
   - Precision: 68.92%
   - Recall: 76.68%
   - AUC Score: 0.7754

2. Confusion Matrix Analysis:
   - True Negatives (Correct Electron Classification): 35,847
   - False Positives (Photons misclassified as Electrons): 16,153
   - False Negatives (Electrons misclassified as Photons): 12,456
   - True Positives (Correct Photon Classification): 39,544

### ROC Curve Analysis
![ROC Curve](https://github.com/user-attachments/assets/7ddf75f7-e2fe-49c2-8030-8a403ad9674c)


The ROC curve demonstrates:
- AUC of 0.7754, indicating good discrimination ability
- Performance significantly better than random classification
- A balanced trade-off between false positive and true positive rates

## Future Improvements

Planned enhancements for the project include:
1. Data Processing:
   - Implementation of data augmentation techniques
   - Dataset cleaning to remove empty or noise-only images
   - Exploration of additional preprocessing methods

2. Model Architecture:
   - Implementation of other architectures (ResNet, EfficientNet)
   - Addition of batch normalization layers
   - Experimentation with different optimization strategies

3. Training Strategy:
   - Training on the full dataset
   - Exploration of class weighting

## Usage

Basic example of using the classifier:

```python
from components import ParticleClassifier, ParticleDataPreprocessor

# Initialize and configure data preprocessor
preprocessor = ParticleDataPreprocessor()
train_dataset, test_dataset, input_shape = preprocessor.preprocess_data(
    electrons_path='data/SingleElectronPt50_IMGCROPS_n249k_RHv1.hdf5',
    photons_path='data/SinglePhotonPt50_IMGCROPS_n249k_RHv1.hdf5',
    sample_fraction=0.2
)

# Create and train the model
classifier = ParticleClassifier(model_type='lenet5')
classifier.train(train_dataset, test_dataset)

# Evaluate performance
metrics = classifier.evaluate(test_dataset)
```

## Requirements

Main dependencies:
- Python >=3.8
- TensorFlow 2.15.0
- NumPy 1.24.3
- h5py 3.10.0
- scikit-learn 1.3.2

A complete list of dependencies is available in the `pyproject.toml` file.

## References

The dataset and methodology are based on research conducted at CERN:
- [CERN Open Data Portal](https://opendata.cern.ch)
- [Particle Classification using Deep Learning](https://iopscience.iop.org/article/10.1088/1742-6596/1085/4/042022/pdf)
