from components import ParticleClassifier, ParticleDataPreprocessor, ParticleVisualizer
from data.DataDownloader import ParticleDataDownloader
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger(__name__)


def compare_model_metrics(models_dict, test_dataset):
    results = {}
    for name, classifier in models_dict.items():
        metrics = classifier.evaluate(test_dataset)
        results[name] = metrics
        print(f"\n{name} Results:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
    return results

if __name__ == "__main__":
    #Download data
    ParticleDataDownloader()

    # Initialize preprocessor
    preprocessor = ParticleDataPreprocessor()
    
    # Preprocess data
    train_dataset, test_dataset, input_shape = preprocessor.preprocess_data(
        electrons_path='./data/SingleElectronPt50_IMGCROPS_n249k_RHv1.hdf5',
        photons_path='./data/SinglePhotonPt50_IMGCROPS_n249k_RHv1.hdf5',
        sample_fraction=0.2,
        batch_size=32
    )

    # Validate the data
    preprocessor.validate_data(train_dataset)

    # lenet5 initialization
    classifier_lenet = ParticleClassifier(model_type='lenet5')

    # Train the model
    classifier_lenet.train(train_dataset, test_dataset)
    classifier_lenet.evaluate(test_dataset)

    classifier_lenet.summary_model()

    classifier_lenet.plot_history()


    classifier_lenet.load_model('lenet.keras')

    visualizer_lenet = ParticleVisualizer(classifier_lenet.model)

    logging.info("\nAnalyzing LeNet-5 performance...")
    pred_lenet, true_lenet = visualizer_lenet.analyze_model_performance(test_dataset)


    # custom model initialization
    classifier_custom = ParticleClassifier(model_type='custom_model')
    classifier_custom.train(train_dataset, test_dataset)
    classifier_custom.evaluate(test_dataset)
    classifier_custom.summary_model()
    classifier_custom.plot_history()

    classifier_custom.load_model('custom.keras')

    visualizer_custom = ParticleVisualizer(classifier_custom.model)

    logging.info("\nAnalyzing custom model performance...")

    pred_custom, true_custom = visualizer_custom.analyze_model_performance(test_dataset)

    # efficientnet initialization
    classifier_efficientnet = ParticleClassifier(model_type='efficientnet')
    classifier_efficientnet.train(train_dataset, test_dataset)
    classifier_efficientnet.evaluate(test_dataset)
    classifier_efficientnet.summary_model()
    classifier_efficientnet.plot_history()

    classifier_efficientnet.load_model('efficientnet.keras')

    visualizer_efficientnet = ParticleVisualizer(classifier_efficientnet.model)

    logging.info("\nAnalyzing EfficientNet performance...")

    pred_efficientnet, true_efficientnet = visualizer_efficientnet.analyze_model_performance(test_dataset)

    models = {
        'LeNet-5': classifier_lenet,
        'custom_model': classifier_custom,
        'efficientnet': classifier_efficientnet
    }

    metrics_comparison = compare_model_metrics(models, test_dataset)

