import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
import tensorflow as tf

class ParticleVisualizer:
    def __init__(self, model, class_names=['Electron', 'Photon']):
        self.model = model
        self.class_names = class_names

    def predict_batch(self, dataset):
        """Make predictions on a TensorFlow dataset"""
        predictions = []
        true_labels = []

        for x_batch, y_batch in dataset:
            batch_predictions = self.model.predict(x_batch, verbose=0)
            predictions.extend(batch_predictions)
            true_labels.extend(y_batch.numpy())

        return np.array(predictions), np.array(true_labels)

    def plot_sample_images(self, dataset, num_images=5):
        """Plot sample images with their predictions"""
        plt.figure(figsize=(15, 3))

        for i, (images, labels) in enumerate(dataset.take(1)):
            predictions = self.model.predict(images[:num_images], verbose=0)

            for j in range(num_images):
                plt.subplot(1, num_images, j + 1)
                img_display = np.sum(images[j].numpy(), axis=-1)
                plt.imshow(img_display, cmap='viridis')

                true_label = self.class_names[int(labels[j][0])]
                pred_label = self.class_names[int(round(predictions[j][0]))]
                color = 'green' if true_label == pred_label else 'red'

                plt.title(f'True: {true_label}\nPred: {pred_label}',
                         color=color)
                plt.axis('off')

        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self, true_labels, predictions, threshold=0.5):
        """Plot confusion matrix"""
        pred_labels = (predictions >= threshold).astype(int)
        cm = confusion_matrix(true_labels, pred_labels)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

    def plot_roc_curve(self, true_labels, predictions):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(true_labels, predictions)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()

    def plot_prediction_distribution(self, predictions, true_labels):
        """Plot distribution of predictions for each class"""
        plt.figure(figsize=(10, 6))

        for i, class_name in enumerate(self.class_names):
            class_preds = predictions[true_labels == i]
            plt.hist(class_preds, bins=50, alpha=0.5,
                    label=f'True {class_name}')

        plt.xlabel('Prediction Score')
        plt.ylabel('Count')
        plt.title('Distribution of Prediction Scores by Class')
        plt.legend()
        plt.show()

    def analyze_model_performance(self, dataset):
        """Comprehensive analysis of model performance"""
        predictions, true_labels = self.predict_batch(dataset)

        print("Generating visualization suite...")

        self.plot_sample_images(dataset)

        self.plot_confusion_matrix(true_labels, predictions)

        self.plot_roc_curve(true_labels, predictions)

        self.plot_prediction_distribution(predictions, true_labels)

        return predictions, true_labels